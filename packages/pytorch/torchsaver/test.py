#!/usr/bin/env python3
import os
import re
import time
import shutil
import requests
import argparse
import torch
print('testing torch-memory-saver...')
import torch_memory_saver

memory_saver = torch_memory_saver.TorchMemorySaver()

# 1. For tensors that wants to be paused, create them within `region`
with memory_saver.region():
    x = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. After `pause`, CUDA memory is released for those tensors.
# For example, check `nvidia-smi`'s memory usage to verify.
memory_saver.pause()

# 3. After `resume`, CUDA memory is re-occupied for those tensors.
memory_saver.resume()
print('torch-memory-saver OK')