import gc
import importlib
import json
import os
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from math import ceil
from typing import Union

import safetensors.torch
import torch
import torch.nn as nn
import tqdm
import yaml
from cachetools import LRUCache
from diffusers import (
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers


class EvictingLRUCache(LRUCache):
    def __init__(self, maxsize, device, getsizeof=None):
        LRUCache.__init__(self, maxsize, getsizeof)
        self.device = device

    def popitem(self):
        key, val = LRUCache.popitem(self)
        key.to("cpu")  # type: ignore
        return key, val

    def __setitem__(self, key, value, *args, **kwargs):
        LRUCache.__setitem__(self, key, value, *args, **kwargs)
        key.to(self.device)  # type: ignore


def move(modul, moe_device, move_fn):
    def move_to_device(
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        memory_format: torch.memory_format = torch.channels_last,
    ):
        def _move(
            module,
            moe: torch.device,
            device: torch.device,
            dtype: torch.dtype = torch.float16,
            memory_format: torch.memory_format = torch.channels_last,
        ):
            if isinstance(module, SparseMoeBlock):
                module.move = move_fn
                module.to(device=moe, dtype=dtype, memory_format=memory_format)  # type: ignore
            else:
                module.to(device=device, dtype=dtype, memory_format=memory_format)
                for child in module.children():
                    _move(child, moe, device, dtype, memory_format)

        for child in modul.children():
            _move(child, moe_device, device, dtype, memory_format)

    return move_to_device


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_forward_hooks(child)


# Inspired from transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock
class SparseMoeBlock(nn.Module):
    def __init__(self, config, experts):
        super().__init__()
        self.hidden_dim = config["hidden_size"]
        self.num_experts = config["num_local_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.move = config.get("move_fn", lambda _: _)
        self.out_dim = config.get("out_dim", self.hidden_dim)

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([deepcopy(exp) for exp in experts])

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, sequence_length, f_map_sz = hidden_states.shape
        hidden_states = hidden_states.view(-1, f_map_sz)

        self.move(self.gate)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        _, selected_experts = torch.topk(
            router_logits.sum(dim=0, keepdim=True), self.top_k, dim=1
        )
        routing_weights = nn.functional.softmax(
            router_logits[:, selected_experts[0]], dim=1, dtype=torch.float
        )

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.out_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Loop over all available experts in the model and perform the computation on each expert
        for i, expert_idx in enumerate(selected_experts[0].tolist()):
            expert_layer = self.experts[expert_idx]

            expert_layer = self.move(expert_layer)

            current_hidden_states = routing_weights[:, i].view(
                batch_size * sequence_length, -1
            ) * expert_layer(hidden_states)

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states = final_hidden_states + current_hidden_states
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, self.out_dim
        )
        return final_hidden_states


def getActivation(activation, name):
    def hook(model, inp, output):
        activation[name] = inp

    return hook


class SegMoEPipeline:
    def __init__(self, config_or_path, **kwargs) -> None:
        """
        Instantiates the SegMoEPipeline. SegMoEPipeline implements the Segmind Mixture of Diffusion Experts, efficiently combining Stable Diffusion and Stable Diffusion Xl models.

        Usage:

        from segmoe import SegMoEPipeline
        pipeline = SegMoEPipeline(config_or_path, **kwargs)

        config_or_path: Path to Config or Directory containing SegMoE checkpoint or HF Card of SegMoE Checkpoint.

        Other Keyword Arguments:
        torch_dtype: Data Type to load the pipeline in. (Default: torch.float16)
        variant: Variant of the Model. (Default: fp16)
        device: Device to load the model on. (Default: cuda)
        on_device_layers: How many layers to keep on device, '-1' for all layers. (Default: -1)
        scheduler: Which scheduler to use for sampling. (Default: DDPMScheduler)
        Other args supported by diffusers.DiffusionPipeline are also supported.

        For more details visit https://github.com/segmind/segmoe.
        """
        self.torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        self.use_safetensors = kwargs.pop("use_safetensors", True)
        self.variant = kwargs.pop("variant", "fp16")
        self.offload_layer = -1
        self.device = kwargs.pop("device", "cuda")
        self.scheduler = kwargs.pop("scheduler", DDPMScheduler)
        self.config: dict

        self.pipe: DiffusionPipeline = None  # type: ignore
        self.offload_cache: EvictingLRUCache = None  # type: ignore

        self.on_device_layers = kwargs.pop("on_device_layers", self.offload_layer)
        self.scheduler_class = self.scheduler

        if os.path.isfile(config_or_path):
            self.load_from_scratch(config_or_path, **kwargs)
        else:
            if not os.path.isdir(config_or_path):
                cached_folder = DiffusionPipeline.download(config_or_path, **kwargs)
            else:
                cached_folder = config_or_path
            unet = self.create_empty(cached_folder)
            unet.load_state_dict(
                safetensors.torch.load_file(
                    f"{cached_folder}/unet/diffusion_pytorch_model.safetensors"
                )
            )
            if self.config.get("type", "sdxl") == "sdxl":
                self.base_cls = StableDiffusionXLPipeline
            elif self.config.get("type", "sdxl") == "sd":
                self.base_cls = StableDiffusionPipeline
            else:
                raise NotImplementedError(
                    "Base class not yet supported, type should be one of ['sd','sdxl]"
                )
            if self.offload_layer != -1:
                unet.to = move(unet, "cpu")  # type: ignore
            self.pipe = self.base_cls.from_pretrained(
                cached_folder,
                unet=unet,
                torch_dtype=self.torch_dtype,
                use_safetensors=self.use_safetensors,
                **kwargs,
            )
            self.pipe.to(self.device)
            self.pipe.unet.to(
                device=self.device,
                dtype=self.torch_dtype,
                memory_format=torch.channels_last,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def scheduler_class(self):
        return self.scheduler

    @scheduler_class.setter
    def scheduler_class(self, value: Union[KarrasDiffusionSchedulers, int, str, Enum]):
        if (isinstance(value, str) and value.isdigit()) or isinstance(
            self.scheduler, (int, KarrasDiffusionSchedulers)
        ):
            if isinstance(value, Enum):
                self.scheduler = value.value
            scheduler = KarrasDiffusionSchedulers(int(value))  # type: ignore
            try:
                self.scheduler = getattr(
                    importlib.import_module("diffusers"), scheduler.name
                )
            except AttributeError:
                pass
        elif isinstance(value, str):
            self.scheduler = getattr(importlib.import_module("diffusers"), value)
        else:
            self.scheduler = value

    @property
    def on_device_layers(self):
        return self.offload_layer

    @on_device_layers.setter
    def on_device_layers(self, value: int):
        self.offload_layer = value
        if self.offload_layer != -1:
            if self.offload_cache is not None:
                list(map(lambda x: x.to("cpu"), self.offload_cache.keys()))
            self.offload_cache = EvictingLRUCache(self.offload_layer, self.device)

            def move_fn(module: nn.Module):
                self.offload_cache[module] = None
                return module

            self.move_fn = move_fn
            offload_device = "cpu"
        else:
            offload_device = self.device
            self.move_fn = lambda _: _
        if self.pipe is not None:
            self.pipe.unet.to = move(self.pipe.unet, offload_device, self.move_fn)  # type: ignore
            self.pipe.unet.to(device=self.device)

    @classmethod
    def download_url(cls, file: str, url: str) -> None:
        os.makedirs(file.split("/")[0], exist_ok=True)
        if not os.path.isfile(file):
            os.system(f"wget {url} -O {file} --content-disposition")

    def load_from_scratch(self, config: str, **kwargs) -> None:  # type: ignore
        # Load Config
        with open(config, "r") as f:
            config: dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = config
        self.num_experts = max(
            self.config.get("num_experts", 1),
            len(self.config.get("experts", [])),
            len(self.config.get("loras", [])),
        )
        self.config["base_model"] = self.config.get(
            "base_model", self.config["experts"][0]["source_model"]
        )
        self.config["num_experts_per_tok"] = num_experts_per_tok = self.config.get(
            "num_experts_per_tok", 1
        )
        self.config["moe_layers"] = moe_layers = self.config.get("moe_layers", "attn")
        self.config["type"] = self.config.get("type", "sdxl")
        if self.config["type"] == "sdxl":
            self.base_cls = StableDiffusionXLPipeline
        elif self.config["type"] == "sd":
            self.base_cls = StableDiffusionPipeline
        else:
            raise NotImplementedError(
                f"Base class {self.config['type']} not supported yet: Type should be one of ['sd', 'sdxl]"
            )

        # Load Base Model
        if self.config["base_model"].startswith(
            "https://civitai.com/api/download/models/"
        ):
            self.download_url("base/model.safetensors", self.config["base_model"])
            self.config["base_model"] = "base/model.safetensors"
            self.pipe = self.base_cls.from_single_file(
                self.config["base_model"], torch_dtype=self.torch_dtype
            )
        elif os.path.isfile(self.config["base_model"]):
            self.pipe = self.base_cls.from_single_file(
                self.config["base_model"],
                torch_dtype=self.torch_dtype,
                use_safetensors=self.use_safetensors,
                **kwargs,
            )
        else:
            try:
                self.pipe = self.base_cls.from_pretrained(
                    self.config["base_model"],
                    torch_dtype=self.torch_dtype,
                    use_safetensors=self.use_safetensors,
                    variant=self.variant,
                    **kwargs,
                )
            except Exception:
                self.pipe = self.base_cls.from_pretrained(
                    self.config["base_model"], torch_dtype=self.torch_dtype, **kwargs
                )
        if self.base_cls == StableDiffusionPipeline:
            self.up_idx_start = 1
            self.up_idx_end = len(self.pipe.unet.up_blocks)
            self.down_idx_start = 0
            self.down_idx_end = len(self.pipe.unet.down_blocks) - 1
        elif self.base_cls == StableDiffusionXLPipeline:
            self.up_idx_start = 0
            self.up_idx_end = len(self.pipe.unet.up_blocks) - 1
            self.down_idx_start = 1
            self.down_idx_end = len(self.pipe.unet.down_blocks)
        self.config["up_idx_start"] = self.up_idx_start
        self.config["up_idx_end"] = self.up_idx_end
        self.config["down_idx_start"] = self.down_idx_start
        self.config["down_idx_end"] = self.down_idx_end

        self.pipe.scheduler = self.scheduler.from_config(self.pipe.scheduler.config)  # type: ignore

        # Load Experts
        experts = []
        positive = []
        negative = []
        if self.config.get("experts", None):
            for i, exp in enumerate(self.config["experts"]):
                positive.append(exp["positive_prompt"])
                negative.append(exp["negative_prompt"])
                if exp["source_model"].startswith(
                    "https://civitai.com/api/download/models/"
                ):
                    try:
                        self.download_url(
                            f"expert_{i}/model.safetensors", exp["source_model"]
                        )
                        exp["source_model"] = f"expert_{i}/model.safetensors"
                        expert = self.base_cls.from_single_file(
                            exp["source_model"], **kwargs
                        ).to(self.device, self.torch_dtype)
                    except Exception as e:
                        print(f"Expert {i} {exp['source_model']} failed to load")
                        print("Error:", e)
                elif os.path.isfile(exp["source_model"]):
                    expert = self.base_cls.from_single_file(
                        exp["source_model"],
                        torch_dtype=self.torch_dtype,
                        use_safetensors=self.use_safetensors,
                        variant=self.variant,
                        **kwargs,
                    )
                    expert.scheduler = self.scheduler.from_config(  # type: ignore
                        expert.scheduler.config
                    )
                else:
                    try:
                        expert = self.base_cls.from_pretrained(
                            exp["source_model"],
                            torch_dtype=self.torch_dtype,
                            use_safetensors=self.use_safetensors,
                            variant=self.variant,
                            **kwargs,
                        )

                        expert.scheduler = self.scheduler.from_config(  # type: ignore
                            expert.scheduler.config
                        )
                    except Exception:
                        expert = self.base_cls.from_pretrained(
                            exp["source_model"], torch_dtype=self.torch_dtype, **kwargs
                        )
                        expert.scheduler = self.scheduler.from_config(  # type: ignore
                            expert.scheduler.config
                        )
                if exp.get("loras", None):
                    for j, lora in enumerate(exp["loras"]):
                        if lora.get("positive_prompt", None):
                            positive[-1] += " " + lora["positive_prompt"]
                        if lora.get("negative_prompt", None):
                            negative[-1] += " " + lora["negative_prompt"]
                        if lora["source_model"].startswith(
                            "https://civitai.com/api/download/models/"
                        ):
                            try:
                                self.download_url(
                                    f"expert_{i}/lora_{i}/pytorch_lora_weights.safetensors",
                                    lora["source_model"],
                                )
                                lora["source_model"] = f"expert_{j}/lora_{j}"
                                expert.load_lora_weights(lora["source_model"])
                                if len(exp["loras"]) == 1:
                                    expert.fuse_lora()
                            except Exception as e:
                                print(
                                    f"Expert{i} LoRA {j} {lora['source_model']} failed to load"
                                )
                                print("Error:", e)
                        else:
                            expert.load_lora_weights(lora["source_model"])
                            if len(exp["loras"]) == 1:
                                expert.fuse_lora()
                experts.append(expert)
        else:
            experts = [deepcopy(self.pipe) for _ in range(self.num_experts)]
        if self.config.get("experts", None):
            if self.config.get("loras", None):
                for i, lora in enumerate(self.config["loras"]):
                    if lora["source_model"].startswith(
                        "https://civitai.com/api/download/models/"
                    ):
                        try:
                            self.download_url(
                                f"lora_{i}/pytorch_lora_weights.safetensors",
                                lora["source_model"],
                            )
                            lora["source_model"] = f"lora_{i}"
                            self.pipe.load_lora_weights(lora["source_model"])
                            if len(self.config["loras"]) == 1:
                                self.pipe.fuse_lora()
                        except Exception as e:
                            print(f"LoRA {i} {lora['source_model']} failed to load")
                            print("Error:", e)
                    else:
                        self.pipe.load_lora_weights(lora["source_model"])
                        if len(self.config["loras"]) == 1:
                            self.pipe.fuse_lora()
        else:
            if self.config.get("loras", None):
                j = []
                n_loras = len(self.config["loras"])
                i = 0
                positive = [""] * len(experts)
                negative = [""] * len(experts)
                while n_loras:
                    n = ceil(n_loras / len(experts))
                    j += [i] * n
                    n_loras -= n
                    i += 1
                for i, lora in enumerate(self.config["loras"]):
                    positive[j[i]] += lora["positive_prompt"] + " "
                    negative[j[i]] += lora["negative_prompt"] + " "
                    if lora["source_model"].startswith(
                        "https://civitai.com/api/download/models/"
                    ):
                        try:
                            self.download_url(
                                f"lora_{i}/pytorch_lora_weights.safetensors",
                                lora["source_model"],
                            )
                            lora["source_model"] = f"lora_{i}"
                            experts[j[i]].load_lora_weights(lora["source_model"])
                            experts[j[i]].fuse_lora()
                        except Exception:
                            print(f"LoRA {i} {lora['source_model']} failed to load")
                    else:
                        experts[j[i]].load_lora_weights(lora["source_model"])
                        experts[j[i]].fuse_lora()

        down_blocks = list(
            map(
                lambda i: (i, ("d", self.pipe.unet.down_blocks[i])),
                range(self.down_idx_start, self.down_idx_end),
            )
        )
        up_blocks = list(
            map(
                lambda i: (i, ("u", self.pipe.unet.up_blocks[i])),
                range(self.up_idx_start, self.up_idx_end),
            )
        )
        self.all_blocks = down_blocks + up_blocks

        config_base = {
            "num_experts_per_tok": num_experts_per_tok,
            "num_local_experts": len(experts),
        }

        # Replace FF and Attention Layers with Sparse MoE Layers
        for i, (t, block) in self.all_blocks:
            for j, attention in enumerate(block.attentions):
                for k, transformer in enumerate(attention.transformer_blocks):
                    if moe_layers != "attn":
                        config = {
                            "hidden_size": next(transformer.ff.parameters()).size()[-1],
                            **config_base,
                        }
                        # FF Layers
                        layers = list(
                            map(
                                lambda expert: deepcopy(
                                    (
                                        expert.unet.down_blocks
                                        if t == "d"
                                        else expert.unet.up_blocks
                                    )[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .ff
                                ),
                                experts,
                            )
                        )
                        transformer.ff = SparseMoeBlock(config, layers)
                    if moe_layers != "ff":
                        ## Attns
                        config = {
                            "hidden_size": transformer.attn1.to_q.weight.size()[-1],
                            **config_base,
                        }
                        layers = list(
                            map(
                                lambda expert: deepcopy(
                                    (
                                        expert.unet.down_blocks
                                        if t == "d"
                                        else expert.unet.up_blocks
                                    )[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_q
                                ),
                                experts,
                            )
                        )
                        transformer.attn1.to_q = SparseMoeBlock(config, layers)

                        layers = list(
                            map(
                                lambda expert: deepcopy(
                                    (
                                        expert.unet.down_blocks
                                        if t == "d"
                                        else expert.unet.up_blocks
                                    )[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_k
                                ),
                                experts,
                            )
                        )
                        transformer.attn1.to_k = SparseMoeBlock(config, layers)

                        layers = list(
                            map(
                                lambda expert: deepcopy(
                                    (
                                        expert.unet.down_blocks
                                        if t == "d"
                                        else expert.unet.up_blocks
                                    )[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_v
                                ),
                                experts,
                            )
                        )
                        transformer.attn1.to_v = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": transformer.attn2.to_q.weight.size()[-1],
                            **config_base,
                        }
                        layers = list(
                            map(
                                lambda expert: deepcopy(
                                    (
                                        expert.unet.down_blocks
                                        if t == "d"
                                        else expert.unet.up_blocks
                                    )[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_q
                                ),
                                experts,
                            )
                        )
                        transformer.attn2.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": transformer.attn2.to_k.weight.size()[-1],
                            "out_dim": transformer.attn2.to_k.weight.size()[0],
                            **config_base,
                        }
                        layers = list(
                            map(
                                lambda expert: deepcopy(
                                    (
                                        expert.unet.down_blocks
                                        if t == "d"
                                        else expert.unet.up_blocks
                                    )[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_k
                                ),
                                experts,
                            )
                        )
                        transformer.attn2.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": transformer.attn2.to_v.weight.size()[-1],
                            "out_dim": transformer.attn2.to_v.weight.size()[0],
                            **config_base,
                        }
                        layers = list(
                            map(
                                lambda expert: deepcopy(
                                    (
                                        expert.unet.down_blocks
                                        if t == "d"
                                        else expert.unet.up_blocks
                                    )[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_v
                                ),
                                experts,
                            )
                        )
                        transformer.attn2.to_v = SparseMoeBlock(config, layers)

        # Routing Weight Initialization
        if self.config.get("init", "hidden") == "hidden":
            gate_params = self.get_gate_params(experts, positive, negative)
            for i, (t, block) in self.all_blocks:
                for j, attention in enumerate(block.attentions):
                    for k, transformer in enumerate(attention.transformer_blocks):
                        if moe_layers != "attn":
                            transformer.ff.gate.weight = nn.Parameter(
                                gate_params[f"{t}{i}a{j}t{k}"]
                            )
                        if moe_layers != "ff":
                            transformer.attn1.to_q.gate.weight = nn.Parameter(
                                gate_params[f"sattnq{t}{i}a{j}t{k}"]
                            )
                            transformer.attn1.to_k.gate.weight = nn.Parameter(
                                gate_params[f"sattnk{t}{i}a{j}t{k}"]
                            )
                            transformer.attn1.to_v.gate.weight = nn.Parameter(
                                gate_params[f"sattnv{t}{i}a{j}t{k}"]
                            )

                            transformer.attn2.to_q.gate.weight = nn.Parameter(
                                gate_params[f"cattnq{t}{i}a{j}t{k}"]
                            )
                            transformer.attn2.to_k.gate.weight = nn.Parameter(
                                gate_params[f"cattnk{t}{i}a{j}t{k}"]
                            )
                            transformer.attn2.to_v.gate.weight = nn.Parameter(
                                gate_params[f"cattnv{t}{i}a{j}t{k}"]
                            )
        self.config["num_experts"] = len(experts)
        remove_all_forward_hooks(self.pipe.unet)
        try:
            del experts
            del expert
        except Exception:
            pass
        # Move Model to Device
        self.pipe.to(self.device)
        self.pipe.unet.to(
            device=self.device,
            dtype=self.torch_dtype,
            memory_format=torch.channels_last,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(self, *args, **kwargs) -> None:
        """
        Inference the SegMoEPipeline.

        Calls diffusers.DiffusionPipeline forward with the keyword arguments. See https://github.com/segmind/segmoe#usage for detailed usage.
        """
        output = self.pipe(*args, **kwargs)  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output

    def create_empty(self, path):
        with open(f"{path}/unet/config.json", "r") as f:
            config = json.load(f)
        self.config = config["segmoe_config"]
        unet: UNet2DConditionModel = UNet2DConditionModel.from_config(config)  # type: ignore
        num_experts_per_tok = self.config["num_experts_per_tok"]
        num_experts = self.config["num_experts"]
        moe_layers = self.config["moe_layers"]
        self.up_idx_start = self.config["up_idx_start"]
        self.up_idx_end = self.config["up_idx_end"]
        self.down_idx_start = self.config["down_idx_start"]
        self.down_idx_end = self.config["down_idx_end"]

        down_blocks = list(
            map(
                lambda i: unet.down_blocks[i],
                range(self.down_idx_start, self.down_idx_end),
            )
        )
        up_blocks = list(
            map(
                lambda i: unet.up_blocks[i],
                range(self.up_idx_start, self.up_idx_end),
            )
        )
        all_blocks = down_blocks + up_blocks

        config_base = {
            "num_experts_per_tok": num_experts_per_tok,
            "num_local_experts": num_experts,
            "move_fn": self.move_fn,
        }

        for block in all_blocks:
            for attention in block.attentions:
                for transformer in attention.transformer_blocks:
                    if moe_layers != "attn":
                        config = {
                            "hidden_size": next(transformer.ff.parameters()).size()[-1],
                            **config_base,
                        }
                        layers = [transformer.ff] * num_experts
                        transformer.ff = SparseMoeBlock(config, layers)
                    if moe_layers != "ff":
                        config = {
                            "hidden_size": transformer.attn1.to_q.weight.size()[-1],
                            **config_base,
                        }
                        layers = [transformer.attn1.to_q] * num_experts
                        transformer.attn1.to_q = SparseMoeBlock(config, layers)

                        layers = [transformer.attn1.to_k] * num_experts
                        transformer.attn1.to_k = SparseMoeBlock(config, layers)

                        layers = [transformer.attn1.to_v] * num_experts
                        transformer.attn1.to_v = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": transformer.attn2.to_q.weight.size()[-1],
                            **config_base,
                        }
                        layers = [transformer.attn2.to_q] * num_experts
                        transformer.attn2.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": transformer.attn2.to_k.weight.size()[-1],
                            "out_dim": transformer.attn2.to_k.weight.size()[0],
                            **config_base,
                        }
                        layers = [transformer.attn2.to_k] * num_experts
                        transformer.attn2.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": transformer.attn2.to_v.weight.size()[-1],
                            "out_dim": transformer.attn2.to_v.weight.size()[0],
                            **config_base,
                        }
                        layers = [transformer.attn2.to_v] * num_experts
                        transformer.attn2.to_v = SparseMoeBlock(config, layers)
        return unet

    def save_pretrained(self, path):
        """
        Save SegMoEPipeline to Disk.

        Usage:
        pipeline.save_pretrained(path)

        Parameters:
        path: Path to Directory to save the model in.
        """
        for param in self.pipe.unet.parameters():
            param.data = param.data.contiguous()
        self.pipe.unet.config["segmoe_config"] = self.config
        self.pipe.save_pretrained(path)
        safetensors.torch.save_file(
            self.pipe.unet.state_dict(),
            f"{path}/unet/diffusion_pytorch_model.safetensors",
        )

    def cast_hook(self, pipe, dicts):
        down_blocks = list(
            map(
                lambda i: (i, ("d", pipe.unet.down_blocks[i])),
                range(self.down_idx_start, self.down_idx_end),
            )
        )
        up_blocks = list(
            map(
                lambda i: (i, ("u", pipe.unet.up_blocks[i])),
                range(self.up_idx_start, self.up_idx_end),
            )
        )
        all_blocks = down_blocks + up_blocks

        for i, (t, block) in all_blocks:
            for j, attention in enumerate(block.attentions):
                for k, transformer in enumerate(attention.transformer_blocks):
                    transformer.ff.register_forward_hook(
                        getActivation(dicts, f"{t}{i}a{j}t{k}")
                    )

                    transformer.attn1.to_q.register_forward_hook(
                        getActivation(dicts, f"sattnq{t}{i}a{j}t{k}")
                    )
                    transformer.attn1.to_k.register_forward_hook(
                        getActivation(dicts, f"sattnk{t}{i}a{j}t{k}")
                    )
                    transformer.attn1.to_v.register_forward_hook(
                        getActivation(dicts, f"sattnv{t}{i}a{j}t{k}")
                    )

                    transformer.attn2.to_q.register_forward_hook(
                        getActivation(dicts, f"cattnq{t}{i}a{j}t{k}")
                    )
                    transformer.attn2.to_k.register_forward_hook(
                        getActivation(dicts, f"cattnk{t}{i}a{j}t{k}")
                    )
                    transformer.attn2.to_v.register_forward_hook(
                        getActivation(dicts, f"cattnv{t}{i}a{j}t{k}")
                    )

    @torch.no_grad
    def get_hidden_states(self, model, positive, negative, average: bool = True):
        intermediate = {}
        self.cast_hook(model, intermediate)
        with torch.no_grad():
            _ = model(positive, negative_prompt=negative, num_inference_steps=25)
        hidden = {}
        for key in intermediate:
            hidden_states = intermediate[key][0][-1]
            if average:
                # use average over sequence
                hidden_states = hidden_states.sum(dim=0) / hidden_states.shape[0]
            else:
                # take last value
                hidden_states = hidden_states[:-1]
            hidden[key] = hidden_states.to(self.device)
        del intermediate
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return hidden

    @torch.no_grad
    def get_gate_params(
        self,
        experts,
        positive,
        negative,
    ):
        gate_vects = {}
        hidden_states = []
        for i, expert in enumerate(tqdm.tqdm(experts, desc="Expert Prompts")):
            expert.to(self.device)
            expert.unet.to(
                device=self.device,
                dtype=self.torch_dtype,
                memory_format=torch.channels_last,
            )
            hidden_states = self.get_hidden_states(expert, positive[i], negative[i])
            del expert
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for h in hidden_states:
                if i == 0:
                    gate_vects[h] = []
                hidden_states[h] /= (
                    hidden_states[h].norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                )
                gate_vects[h].append(hidden_states[h])
        for h in hidden_states:
            gate_vects[h] = torch.stack(
                gate_vects[h], dim=0
            )  # (num_expert, num_layer, hidden_size)
            gate_vects[h].permute(1, 0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return gate_vects
