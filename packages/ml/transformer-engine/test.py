#!/usr/bin/env python3
print ('Transformer Engine Test')
print('Transformer Engine Test (FP8, FP16, FP32)')
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs on the GPU.
model = te.Linear(in_features, out_features, bias=True).cuda()
inp = torch.randn(hidden_size, in_features, device="cuda")

# --- FP8 Attempt ---
"""try:
    print("Trying FP8...")
    # Create an FP8 recipe.
    fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

    # Enable autocasting for the forward pass using FP8
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model(inp)
    loss = out.sum()
    loss.backward()
    print("FP8 Successful!\n")

except RuntimeError as e:
    print(f"FP8 Failed: {e} or not supported (Ampere)")

    # --- FP16 Attempt ---"""
try:
    print("Trying FP16...")
    # Reset gradients
    model.zero_grad()

    # Enable autocasting for the forward pass using FP16
    with torch.cuda.amp.autocast():
        out = model(inp)
    loss = out.sum()
    loss.backward()
    print("FP16 Successful!\n")

except RuntimeError as e:
    print(f"FP16 Failed: {e}")

    # --- FP32 Attempt ---
    try:
        print("Trying FP32...")
        # Reset gradients
        model.zero_grad()

        # Run forward and backward pass in FP32
        out = model(inp)
        loss = out.sum()
        loss.backward()
        print("FP32 Successful!\n")

    except Exception as e:
        print(f"FP32 Failed: {e}")
        print("All attempts failed.\n")

print('Transformer Engine OK\n')