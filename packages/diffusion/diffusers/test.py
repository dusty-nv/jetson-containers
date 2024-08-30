#!/usr/bin/env python3
print('testing diffusers...')

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
# pipeline("An image of a squirrel in Picasso style").images[0]

print('diffusers OK\n')

