#!/usr/bin/env python3

from huggingface_hub import hf_hub_download
from model_zigma import ZigMa
import torch
hf_hub_download(
        repo_id="taohu/zigma",
        filename="faceshq1024_0090000.pt",
        local_dir="./checkpoints",
    )


img_dim = 32
in_channels = 3

model = ZigMa(
in_channels=in_channels,
embed_dim=640,
depth=18,
img_dim=img_dim,
patch_size=1,
has_text=True,
d_context=768,
n_context_token=77,
device="cuda",
scan_type="zigzagN8",
use_pe=2,
)

x = torch.rand(10, in_channels, img_dim, img_dim).to("cuda")
t = torch.rand(10).to("cuda")
_context = torch.rand(10, 77, 768).to("cuda")
o = model(x, t, y=_context)
print(o.shape)

print('zigma OK\n')