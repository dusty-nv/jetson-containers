#!/usr/bin/env python3
import gc
import torch

def clear_memory():
    """Free GPU and CPU memory before running SGLang."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        gc.collect()
        print("✅ Memory cleared")
    except Exception as e:
        print(f"⚠️ Memory cleanup failed: {e}")

print('testing SGLang...')
clear_memory()  # <-- Clean before anything else

import sglang as sgl

print(f"SGLang version: {sgl.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

print('SGLang OK\n')
