#!/usr/bin/env python3

print("Testing FlexPrefill...")
import torch
from flex_prefill import flex_prefill_attention

B, N, H, D = 1, 64000, 32, 64
gamma = 0.9
tau = 0.1

q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, N, H // 4, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, N, H // 4, D, device="cuda", dtype=torch.bfloat16)

flex_prefill_output = flex_prefill_attention(
    q,
    k,
    v,
    gamma,
    tau,
    min_budget=512,
    max_budget=None,
)

print(flex_prefill_output.shape)
print('FlexPrefill OK\n')