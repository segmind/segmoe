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

### SDXL Turbo

To use SDXL Turbo style models, just change the scheduler to DPMSolverMultistepScheduler.

Example Config:

```yaml
base_model: stablediffusionapi/realvisxl-v30-turbo
num_experts: 3
moe_layers: all
num_experts_per_tok: 1
experts:
  - source_model: stablediffusionapi/turbovision_xl
    positive_prompt: "aesthetic, cinematic, hands, portrait, photo, illustration, 8K, hyperdetailed, origami, man, woman, supercar"
    negative_prompt: "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"
  - source_model: stablediffusionapi/realvisxl-v30-turbo
    positive_prompt: "cinematic, portrait, photograph, instagram, fashion, movie, macro shot, 8K, RAW, hyperrealistic, ultra realistic,"
    negative_prompt: "(octane render, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured"
  - source_model: Lykon/dreamshaper-xl-turbo 
    positive_prompt: "minimalist, illustration, award winning art, painting, impressionist, comic, colors, sketch, pencil drawing,"
    negative_prompt: "Compression artifacts, bad art, worst quality, low quality, plastic, fake, bad limbs, conjoined, featureless, bad features, incorrect objects, watermark, ((signature):1.25), logo"
```

Usage:

```python

from segmoe import SegMoEPipeline

pipeline = SegMoETurboPipeline("turbo_config.yaml", device = "cuda")
pipeline.pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.pipe.scheduler.config)

prompt = "cinematic photo (art by Mathias Goeritz:0.9) , photograph, Lush Girlfriend, looking at the camera smiling, Rich ginger hair, Winter, tilt shift, Horror, specular lighting, film grain, Samsung Galaxy, F/5, (cinematic still:1.2), freckles . 35mm photograph, film, bokeh, professional, 4k, highly detailed"

image = pipe(prompt=prompt, num_inference_steps=6, guidance_scale=2).images[0]  

image.save("image.png")
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
