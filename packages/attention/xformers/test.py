#!/usr/bin/env python3
print('testing xformers...')

import xformers
import xformers.info

xformers.info.print_info()

from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

with torch.inference_mode():
    sample = pipe("a small cat")

# optional: You can disable it via
# pipe.disable_xformers_memory_efficient_attention()

print('xformers OK\n')

