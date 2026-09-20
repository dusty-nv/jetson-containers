#!/usr/bin/env python3
import time

import torch

import lerobot
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

print(lerobot.available_extras)
print('lerobot version:', lerobot.__version__)

# Smoke test that actually touches torch/CUDA and a real policy, instead of
# just checking a static module attribute. No checkpoint and no hub download,
# so it still works in an air-gapped/CI build.
assert torch.cuda.is_available(), "CUDA not available inside the container"

cfg = ACTConfig(
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.top": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 96, 96)
        ),
    },
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    pretrained_backbone_weights=None,  # else torchvision fetches ResNet18 weights on construction
    chunk_size=10,
    n_action_steps=10,
    dim_model=128,
    n_heads=4,
    dim_feedforward=256,
    n_encoder_layers=1,
    n_decoder_layers=1,
    use_vae=False,
)
policy = ACTPolicy(cfg).to("cuda").eval()

batch = {
    "observation.state": torch.randn(1, 6, device="cuda"),
    "observation.images.top": torch.rand(1, 3, 96, 96, device="cuda"),
}
with torch.no_grad():
    policy.predict_action_chunk(batch)  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    chunk = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1e3

assert chunk.shape == (1, cfg.chunk_size, 6), chunk.shape
assert torch.isfinite(chunk).all(), "non-finite actions"
print(
    f"ACT forward {dt:.1f} ms | peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"
)
