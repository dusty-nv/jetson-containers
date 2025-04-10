#!/usr/bin/env python3
# https://github.com/dusty-nv/openvla/blob/main/vla-scripts/extern/verify_openvla.py
import torch

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

model="openvla/openvla-7b"
print('loading', model)

processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)

vla = AutoModelForVision2Seq.from_pretrained(
    model, 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

print(vla.config)

# Grab image input & format prompt
image = Image.open("/data/images/lake.jpg").convert('RGB')
prompt = "In: What action should the robot take to stop?\nOut:"

print('prompt:', prompt)

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
print('inputs:', list(inputs.keys()))
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# Execute...
print('action:', action)
