# SegMoE: Segmind Mixture of Experts

A Framework to combine multiple Stable Diffusion XL models into a mixture of experts model. Functions simialar to [mergekit](https://github.com/cg123/mergekit)'s mixtral branch but for Stable Diffusion XL models.

## Installation

```bash
pip install segmoe
```

## Usage

```python
from segmoe import SegMoEPipeline

pipeline = SegMoEPipeline("segmind/SegMoE-v0", device = "cuda")

prompt = "cosmic canvas,  orange city background, painting of a chubby cat"
negative_prompt = "nsfw, bad quality, worse quality"
img = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1024,
    width=1024,
    num_inference_steps=25,
    guidance_scale=7.5,
).images[0]
img.save(f"image.png")
```

## Creating your Own Model

Create a yaml config file, config.yaml, with the following structure:

```yaml
base_model: Base Model Path, Model Card or CivitAI Download Link
num_experts: Number of experts to use
moe_layers: Type of Layers to Mix (can be "ff", "attn" or "all"). Defaults to "attn"
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

For detailed information on how to create a config file, please refer to the [Config Parameters](#config-parameters)

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

## Advantages
+ Benefits from The Knowledge of Several Finetuned Experts
+ Training Free
+ Better Adaptability to Data

## Limitations
+ This is a very early version of the framework and is not yet optimized for speed.
+ The framework is not yet optimized for memory usage.

## Research Roadmap
- [*] Optimize for Speed
- [ ] Optimize for Memory Usage
- [ ] Add Support for LoRAs
- [ ] Add Support for More Models
- [ ] Add Support for Training

## Config Parameters

### Base Model

The base model is the model that will be used to generate the initial image. It can be a Huggingface model card, a CivitAI model download link. 

### Number of Experts

The number of experts to use in the mixture of experts model. The number of experts must be greater than 1.

### MOE Layers

The type of layers to mix. Can be "ff", "attn" or "all". Defaults to "attn"

### Experts

The experts are the models that will be used to generate the final image. Each expert must have a source model, a positive prompt and a negative prompt. The source model can be a Huggingface model card, a CivitAI model download link. The positive prompt and negative prompt are the prompts that will be used to compute the gate weights for each expert.

## Citation

```bibtex
@misc{segmoe,
  author = {Yatharth Gupta, Vishnu V Jaddipal, Harish Prabhala},
  title = {SegMoE: Segmind Mixture of Experts},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/segmind/segmoe}}
}
```
