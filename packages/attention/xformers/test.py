#!/usr/bin/env python3
print("testing xformers with a tiny model…")

import os, torch, xformers, xformers.ops
from diffusers import DiffusionPipeline

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

device_name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"{torch.__version__} {device_name} (sm_{cap[0]}{cap[1]})")
print(f"xformers {xformers.__version__}")

pipe = DiffusionPipeline.from_pretrained(
    "segmind/tiny-sd",
    torch_dtype=torch.float16,
    use_safetensors=False,
    low_cpu_mem_usage=False,
    device_map=None,
).to("cuda")

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers memory-efficient attention enabled")
except NotImplementedError:
    print(f"xformers attention operators not available for sm_{cap[0]}{cap[1]} — running with default attention")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    image = pipe("a small cat", num_inference_steps=20, guidance_scale=7.0, height=256, width=256).images[0]

out_path = "/data/images/tiny_cat.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
image.save(out_path)
print("xformers + tiny-sd OK ->", out_path)
