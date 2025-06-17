#!/usr/bin/env python3
print('testing mlstm_kernels...')

import torch
# directly import mLSTMexp TFLA kernel
from mlstm_kernels.torch.chunkwise.triton_xl_chunk import mlstm_chunkwise__xl_chunk

# run the kernel
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
B = 2
S = 512
DHQK = 128
DHHV = 256
NH = 4

torch.manual_seed(1)
matQ = torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE)
matK = torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE)
matV = torch.randn((B, NH, S, DHHV), dtype=DTYPE, device=DEVICE)
vecI = torch.randn((B, NH, S), dtype=DTYPE, device=DEVICE)
vecF = 3.0 + torch.randn((B, NH, S), dtype=DTYPE, device=DEVICE)

matH1 = mlstm_chunkwise__xl_chunk(
    q=matQ, k=matK, v=matV, i=vecI, f=vecF, return_last_states=False, chunk_size=256
)
print('mlstm_kernels OK\n')