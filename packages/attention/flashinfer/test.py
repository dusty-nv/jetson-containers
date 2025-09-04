#!/usr/bin/env python3
import gc
import torch

def clear_memory():
    """Free GPU and CPU memory before running FlashInfer ops."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        gc.collect()
        print("✅ Memory cleared")
    except Exception as e:
        print(f"⚠️ Memory cleanup failed: {e}")

print("Testing FlashInfer...")
clear_memory()  # <-- clean first

import flashinfer
print('FlashInfer version', flashinfer.__version__)

kv_len = 2048
num_kv_heads = 32
head_dim = 128

k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)

# decode attention
num_qo_heads = 32
q = torch.randn(num_qo_heads, head_dim).half().to(0)

o = flashinfer.single_decode_with_kv_cache(q, k, v)  # without RoPE
o_rope_on_the_fly = flashinfer.single_decode_with_kv_cache(
    q, k, v, pos_encoding_mode="ROPE_LLAMA"
)  # with LLaMA-style RoPE

# append attention
append_qo_len = 128
q = torch.randn(append_qo_len, num_qo_heads, head_dim).half().to(0)
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)  # causal, no RoPE
o_rope_on_the_fly = flashinfer.single_prefill_with_kv_cache(
    q, k, v, causal=True, pos_encoding_mode="ROPE_LLAMA"
)  # causal, with RoPE

# prefill attention
qo_len = 2048
q = torch.randn(qo_len, num_qo_heads, head_dim).half().to(0)
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False)  # no causal mask

print('FlashInfer OK\n')
