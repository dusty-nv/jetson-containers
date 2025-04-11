#!/usr/bin/env python3
import os
import time
import logging
import argparse

#logging.basicConfig(level=logging.DEBUG)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import requests

#jax.print_environment_info()

# https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

#jax.config.update("jax_enable_x64", False)

OCTO_MODELS = [
    'rail-berkeley/octo-small',
    'rail-berkeley/octo-small-1.5',
    'rail-berkeley/octo-base',
    'rail-berkeley/octo-base-1.5',
]

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=OCTO_MODELS[-1], help="path to model checkpoints or HuggingFace repo {OCTO_MODELS}")
args = parser.parse_args()
                
# load the model
from octo.model.octo_model import OctoModel

print(f"Loading {args.model}")
model = OctoModel.load_pretrained(f'hf://{args.model}')
print(model.get_pretty_spec())

IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
img = np.array(Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256)))

# add batch + time horizon 1
img = img[np.newaxis,np.newaxis,...]
observation = {"image_primary": img, "timestep_pad_mask": np.array([[True]])}
task = model.create_tasks(texts=["pick up the fork"])

# benchmark performance
print(f"Running octo inference")

for n in range(20):
    time_begin = time.perf_counter()
    action = model.sample_actions(
        observation, 
        task, 
        unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"], 
        rng=jax.random.PRNGKey(0)
    )
    time_elapsed = time.perf_counter() - time_begin
    #print(action)  # [batch, action_chunk, action_dim]
    print(f"{os.path.basename(args.model)}  frame={n}  latency={time_elapsed*1000:.2f} ms  action_dims={action.shape}")
    

