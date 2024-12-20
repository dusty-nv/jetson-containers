#!/usr/bin/env python3
print('testing sana...')

import torch
from app.sana_pipeline import sanaPipeline
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = sanaPipeline("configs/sana_config/1024ms/sana_1600M_img1024.yaml")
sana.from_pretrained("hf://Efficient-Large-Model/sana_1600M_1024px/checkpoints/sana_1600M_1024px.pth")
prompt = 'a cyberpunk cat with a neon sign that says "sana"'

image = sana(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=5.0,
    pag_guidance_scale=2.0,
    num_inference_steps=18,
    generator=generator,
)
save_image(image, 'output/sana.png', nrow=1, normalize=True, value_range=(-1, 1))

print('sana OK\n')