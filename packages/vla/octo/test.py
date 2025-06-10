#!/usr/bin/env python3
import os
import logging

logging.basicConfig(level=logging.DEBUG)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
# https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

print('\nimport octo')
from octo.model.octo_model import OctoModel
print('octo OK')
