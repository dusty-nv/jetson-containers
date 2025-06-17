#!/usr/bin/env python3
print('testing xlstm...')

import torch
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

# configure the model with TFLA Triton kernels
xlstm_config = xLSTMLargeConfig(
    embedding_dim=512,
    num_heads=4,
    num_blocks=6,
    vocab_size=2048,
    return_last_states=True,
    mode="inference",
    chunkwise_kernel="chunkwise--triton_xl_chunk", # xl_chunk == TFLA kernels
    sequence_kernel="native_sequence__triton",
    step_kernel="triton",
)
# instantiate the model
xlstm = xLSTMLarge(xlstm_config)
xlstm = xlstm.to("cuda")
# create inputs
input = torch.randint(0, 2048, (3, 256)).to("cuda")
# run a forward pass
out = xlstm(input)
out.shape[1:] == (256, 2048)

print('xlstm OK\n')