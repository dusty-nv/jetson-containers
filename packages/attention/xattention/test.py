#!/usr/bin/env python3
print("Testing xattention...")
import torch
from xattn.src.Xattention import Xattention_prefill

bsz = 1
heads = 32
seq_len = 1024
dim = 128
q = torch.randn((bsz,heads,seq_len,dim),dtype=torch.bfloat16).to("cuda")
k = torch.randn((bsz,heads,seq_len,dim),dtype=torch.bfloat16).to("cuda")
v = torch.randn((bsz,heads,seq_len,dim),dtype=torch.bfloat16).to("cuda")

attention_output = Xattention_prefill(query_states=q,key_states=k,value_states=v,stride=16,block_size=128,use_triton=True,chunk_size=2048)

print('xattention OK\n')
