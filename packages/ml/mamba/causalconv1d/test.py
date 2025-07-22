#!/usr/bin/env python3
import torch
from causal_conv1d import causal_conv1d_fn
import causal_conv1d_cuda

batch, dim, seq, width = 10, 5, 17, 4
x = torch.zeros((batch, dim, seq)).to('cuda')
weight = torch.zeros((dim, width)).to('cuda')
bias = torch.zeros((dim, )).to('cuda')

causal_conv1d_fn(x, weight, bias, None)
conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, None, None, torch.zeros_like(x), None, True)

print('causal_conv1d OK\n')