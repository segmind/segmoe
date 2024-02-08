# SegMoE: Segmind Mixture of Diffusion Experts

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62f8ca074588fe31f4361dae/TJTQyN9tav94fVcvpZGq8.png)

SegMoE is a powerful framework for dynamically combining Stable Diffusion Models into a Mixture of Experts within minutes without training. The framework allows for creation of larger models on the fly which offer larger knowledge, better adherence and better image quality. It is inspired by [mergekit](https://github.com/cg123/mergekit)'s mixtral branch but for Stable Diffusion models.

## Installation

```bash
pip install segmoe
```

## Usage

### Load Checkpoint from Hugging Face

We release 3 merges on Hugging Face, 

- [SegMoE 2x1](https://huggingface.co/segmind/SegMoE-2x1-v0) has two expert models.
- [SegMoE 4x2](https://huggingface.co/segmind/SegMoE-4x2-v0) has four expert models.
- [SegMoE SD 4x2](https://huggingface.co/segmind/SegMoE-sd-4x2-v0) has four Stable Diffusion 1.5 expert models.

They can be loaded as follows:

```python
from segmoe import SegMoEPipeline

pipeline = SegMoEPipeline("segmind/SegMoE-4x2-v0", device = "cuda")

prompt = "cosmic canvas, orange city background, painting of a chubby cat"
negative_prompt = "nsfw, bad quality, worse quality"
img = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1024,
    width=1024,
    num_inference_steps=25,
    guidance_scale=7.5,
).images[0]
img.save("image.png")
```
## Comparison 

The Prompt Understanding seems to improve as shown in the images below. From Left to Right SegMoE-2x1-v0, SegMoE-4x2-v0, Base Model ([RealVisXL_V3.0](https://huggingface.co/SG161222/RealVisXL_V3.0))

![image](https://github.com/segmind/segmoe/assets/95569637/bcdc1b11-bbf5-4947-b6bb-9f745ff0c040)

<div align="center">three green glass bottles</div>
<br>

![image](https://github.com/segmind/segmoe/assets/95569637/d50e2af0-66d2-4112-aa88-bd4df88cbd5e)

<div align="center">panda bear with aviator glasses on its head</div>
<br>

![image](https://github.com/segmind/segmoe/assets/95569637/aba2954a-80c2-428a-bf76-0a70a5e03e9b)

<div align="center">the statue of Liberty next to the Washington Monument</div>



## Creating your Own Model

Create a yaml config file, config.yaml, with the following structure:

```yaml
base_model: Base Model Path, Model Card or CivitAI Download Link
num_experts: Number of experts to use
moe_layers: Type of Layers to Mix (can be "ff", "attn" or "all"). Defaults to "attn"
num_experts_per_tok: Number of Experts to use 
type: Type of the individual models (can be "sd" or "sdxl"). Defaults to "sdxl"
experts:
  - source_model: Expert 1 Path, Model Card or CivitAI Download Link
    positive_prompt: Positive Prompt for computing gate weights
    negative_prompt: Negative Prompt for computing gate weights
  - source_model: Expert 2 Path, Model Card or CivitAI Download Link
    positive_prompt: Positive Prompt for computing gate weights
    negative_prompt: Negative Prompt for computing gate weights
  - source_model: Expert 3 Path, Model Card or CivitAI Download Link
    positive_prompt: Positive Prompt for computing gate weights
    negative_prompt: Negative Prompt for computing gate weights
  - source_model: Expert 4 Path, Model Card or CivitAI Download Link
    positive_prompt: Positive Prompt for computing gate weights
    negative_prompt: Negative Prompt for computing gate weights
```

Any number of models can be combined, An Example config can be found [here](./segmoe_config_4x2.yaml). For detailed information on how to create a config file, please refer to the [Config Parameters](#config-parameters)

**Note**
Both Huggingface Models and CivitAI Models are supported. For CivitAI models, paste the download link of the model, For Example: "https://civitai.com/api/download/models/239306"

Then run the following command:

```bash
segmoe config.yaml segmoe_v0
```

This will create a folder called segmoe_v0 with the following structure:

```bash
├── model_index.json
├── scheduler
│   └── scheduler_config.json
├── text_encoder
│   ├── config.json
│   └── model.safetensors
├── text_encoder_2
│   ├── config.json
│   └── model.safetensors
├── tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── tokenizer_2
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└──vae
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

Alternatively, you can also use the following command to create a mixture of experts model:

```python

from segmoe import SegMoEPipeline

pipeline = SegMoEPipeline("config.yaml", device="cuda")

pipeline.save_pretrained("segmoe_v0")
```

### Push to Hub

The Model can be pushed to the hub via the huggingface-cli

```bash 
huggingface-cli upload segmind/segmoe_v0 ./segmoe_v0
```

Detailed usage can be found [here](https://huggingface.co/docs/huggingface_hub/guides/upload)

### SDXL Turbo

To use SDXL Turbo style models, just change the scheduler to DPMSolverMultistepScheduler. Example config can be found [here](./segmoe_config_turbo.yaml)

Usage:

```python

from segmoe import SegMoEPipeline

pipeline = SegMoETurboPipeline("segomoe_config_turbo.yaml", device = "cuda")
pipeline.pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.pipe.scheduler.config)

prompt = "cosmic canvas, orange city background, painting of a chubby cat"

image = pipe(prompt=prompt, num_inference_steps=6, guidance_scale=2).images[0]  

image.save("image.png")
```
### Stable Diffusion 1.5 Support

Stable Diffusion 1.5 Models are also supported and work natively. Example config can be found [here](./segmoe_config_sd.yaml)

**Note:** Stable Diffusion 1.5 Models can be combined with other SD1.5 Models only.

## Memory Requirements

- SDXL 2xN : 20GB
- SDXL 4xN : 25GB
- SD1.5 4xN : 7GB

## Advantages
+ Benefits from The Knowledge of Several Finetuned Experts
+ Training Free
+ Better Adaptability to Data
+ Model Can be upgraded by using a better finetuned model as one of the experts.

## Limitations
+ Though the Model improves upon the fidelity of images as well as adherence, it does not be drastically better than any one expert without training and relies on the knowledge of the experts.
+ This is not yet optimized for speed.
+ The framework is not yet optimized for memory usage.

## Research Roadmap
- [ ] Optimize for Speed
- [ ] Optimize for Memory Usage
- [ ] Add Support for LoRAs
- [ ] Add Support for More Models
- [ ] Add Support for Training

## Config Parameters

### Base Model

The base model is the model that will be used to generate the initial image. It can be a Huggingface model card, a CivitAI model download link or a local path to a safetensors file. 

### Number of Experts

The number of experts to use in the mixture of experts model. The number of experts must be greater than 1. The Number of experts can be anything greater than 2 as long as the GPU fits it.

### MOE Layers

The type of layers to mix. Can be "ff", "attn" or "all". Defaults to "attn". "ff" merges only the feedforward layers, "attn" merges only the attention layers and "all" merges all layers.

### Type

The type of the models to mix. Can be "sd" or "sdxl". Defaults to "sdxl".

### Experts

The Experts are the models that will be used to generate the final image. Each expert must have a source model, a positive prompt and a negative prompt. The source model can be a Huggingface model card, a CivitAI model download link or a local path to a safetensors file. The positive prompt and negative prompt are the prompts that will be used to compute the gate weights for each expert and impact the quality of the final model, choose these carefully.

## Citation

```bibtex
@misc{segmoe,
  author = {Yatharth Gupta, Vishnu V Jaddipal, Harish Prabhala},
  title = {SegMoE: Segmind Mixture of Diffusion Experts},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/segmind/segmoe}}
}
```
