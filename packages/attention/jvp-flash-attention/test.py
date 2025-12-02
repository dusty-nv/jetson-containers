#!/usr/bin/env python3
import gc
import torch

def clear_memory():
    """Free GPU and CPU memory before running FlashAttention benchmarks."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        gc.collect()
        print("✅ Memory cleared")
    except Exception as e:
        print(f"⚠️ Memory cleanup failed: {e}")

print("Testing JVP-FlashAttention...")
clear_memory()  # <-- clean first

from jvp_flash_attention.jvp_attention import JVPAttn, attention as jvp_attention
print('JVP-FlashAttention OK\n')
