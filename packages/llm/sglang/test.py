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
from sglang.check_env import check_env

check_env()

print('SGLang OK\n')
