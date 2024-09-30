#!/usr/bin/env python3
import os
import time
import logging

#logging.basicConfig(level=logging.DEBUG)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np

jax.print_environment_info()

# https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

#jax.config.update("jax_enable_x64", False)

# load the model
from crossformer.model.crossformer_model import CrossFormerModel
model = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer")
print(model.get_pretty_spec())

# create a random image
img = np.random.randint(0, 255, size=(224, 224, 3))

# add batch and observation history dimension (CrossFormer accepts a history of up to 5 time-steps)
img = img[None, None]

# our bimanual training data has an overhead view and two wrist views
observation = {
    "image_high": img,
    "image_left_wrist": img,
    "image_right_wrist": img,
    "timestep_pad_mask": np.array([[True]]),
}

# create a task dictionary for a language task
task = model.create_tasks(texts=["uncap the pen"])

# benchmark performance
print(f"Running crossformer inference")

for n in range(20):
    time_begin = time.perf_counter()
    action = model.sample_actions(observation, task, head_name="bimanual", rng=jax.random.PRNGKey(0))
    time_elapsed = time.perf_counter() - time_begin
    #print(action)  # [batch, action_chunk, action_dim]
    print(f"crossformer  frame={n}  latency={time_elapsed*1000:.2f} ms  action_dims={action.shape}")

'''
import jax.experimental.jax2tf as jax2tf
import tensorflow as tf
import tf2onnx

print("converting JAX -> TF")
tf_fn = tf.function(jax2tf.convert(model.sample_actions, enable_xla=False))

tf_args = [tf.TensorSpec(jnp.shape(x), jnp.result_type(x)) for x in [img, task["pad_mask_dict"]["language_instruction"]]]  # pyright: ignore
print("converting TF -> ONNX")
onnx_fn = tf2onnx.convert.from_function(tf_fn, input_signature=tf_args)
print("onnx", onnx_fn)
'''
