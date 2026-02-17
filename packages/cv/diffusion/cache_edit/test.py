#!/usr/bin/env python3
print('testing cache_dit...')
import torch
from diffusers import FluxPipeline
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

print('cache_dit OK\n')
