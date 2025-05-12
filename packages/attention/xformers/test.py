#!/usr/bin/env python3
print("testing xformers with a tiny model…")

import xformers, xformers.info
xformers.info.print_info()

from diffusers import DiffusionPipeline
import torch

# <— swap the model id here
pipe = DiffusionPipeline.from_pretrained(
    "segmind/tiny-sd",          # ~770 MB vs. ~4 GB for SD-1.5
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

with torch.inference_mode():
    image = pipe("a small cat").images[0]
    image.save("tiny_cat.png")

print("xformers + tiny-sd OK ✔")
