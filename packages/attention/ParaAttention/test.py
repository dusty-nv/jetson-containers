#!/usr/bin/env python3

print("Testing ParaAttention...")
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

print('ParaAttention OK\n')
