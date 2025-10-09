#!/usr/bin/env python3
print("testing xformers with a tiny model…")

import os, torch
from diffusers import DiffusionPipeline

# modern allocator env
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

print(torch.__version__, torch.cuda.get_device_name(0))

pipe = DiffusionPipeline.from_pretrained(
    "segmind/tiny-sd",
    torch_dtype=torch.float16,  # your diffusers expects torch_dtype=
    use_safetensors=False,      # tiny-sd uses .bin files
    low_cpu_mem_usage=False,    # stop offload_state_dict path
    device_map=None,            # avoid accelerate auto-offload
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    image = pipe("a small cat", num_inference_steps=20, guidance_scale=7.0, height=256, width=256).images[0]

out_path = "/data/images/tiny_cat.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
image.save(out_path)
print("xformers + tiny-sd OK ✔ ->", out_path)
