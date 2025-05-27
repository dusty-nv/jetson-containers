import dataclasses

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import jax

# Configure JAX matrix multiplication precision
# Available options:
# - 'default': Uses device default precision
# - 'float32': Full float32 precision (highest accuracy)
# - 'float16': Half precision (faster, lower accuracy)
# - 'bfloat16': Brain floating point (good balance)
# - 'tensorfloat32': TF32 on Ampere+ GPUs (speed/accuracy tradeoff)
jax.config.update('jax_default_matmul_precision', 'default')

import sys, os, time
import numpy as np

# Set seeds to reduce randomness (if supported)
np.random.seed(42)
jax_key = jax.random.PRNGKey(42)

from ctypes import cdll

NUM_RUNS = 20

config = _config.get_config("pi0_droid")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_droid")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = droid_policy.make_droid_example()
for k,v in example.items():
    print(f"{k}: {v.shape if isinstance(v, np.ndarray) else v}")
result = policy.infer(example)
print("Actions shape:", result["actions"].shape)

# Load the CUDA runtime library
libcudart = cdll.LoadLibrary('libcudart.so')

libcudart.cudaProfilerStart()

runtimes = []
for i in range(NUM_RUNS):
    start_time = time.time()
    result = policy.infer(example)
    end_time = time.time()
    runtimes.append(end_time - start_time)

libcudart.cudaProfilerStop()

print("====================================================")
print("Walltime (host to host) results")
print(f"Fastest execution: {min(runtimes) * 1000:.3f} ms")
print(f"Slowest execution: {max(runtimes) * 1000:.3f} ms")
print(f"Median execution: {np.median(runtimes) * 1000:.3f} ms")
print("====================================================")
