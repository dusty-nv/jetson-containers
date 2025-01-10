#!/usr/bin/env python3
print ('Transformer Engine Test')
"""import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs.
model = te.Linear(in_features, out_features, bias=True)
inp = torch.randn(hidden_size, in_features, device="cuda")

# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

# Enable autocasting for the forward pass
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out = model(inp)

loss = out.sum()
loss.backward()"""

# AMPERE not has fp8 support
#!/usr/bin/env python3
print('Transformer Engine Test (FP16 with Ampere)')
import torch
import transformer_engine.pytorch as te

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs on the GPU.
model = te.Linear(in_features, out_features, bias=True).cuda()
inp = torch.randn(hidden_size, in_features, device="cuda")

# Enable autocasting for the forward pass using FP16
with torch.cuda.amp.autocast():
    out = model(inp)

loss = out.sum()
loss.backward()

print('Transformer Engine with FP16 OK\n')

print('Transformer Engine OK\n')