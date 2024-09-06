#!/usr/bin/env python3
print('Testing JAX...')

import jax
import jax.numpy as jnp
from jax import random
import os

# Print JAX version and CUDA device information
print('JAX version: ' + str(jax.__version__))
print('CUDA devices: ' + str(jax.devices()))

# Fail if CUDA isn't available
assert len(jax.devices()) > 0, 'No CUDA devices found'

# Check that version can be parsed
from packaging import version

print('PACKAGING_VERSION=' + str(version.parse(jax.__version__)))
print('JAX_CUDA_ARCH_LIST=' + os.environ.get('JAX_CUDA_ARCH_LIST', 'None') + '\n')

# Quick CUDA tensor test
key = random.PRNGKey(0)
a = jnp.zeros(2)
print('Tensor a = ' + str(a))

b = random.normal(key, (2,))
print('Tensor b = ' + str(b))

c = a + b
print('Tensor c = ' + str(c))

# LAPACK test
print('Testing LAPACK (via jax.numpy.linalg)...')
a = random.normal(key, (4, 4))
b = random.normal(key, (4, 4))

x = jnp.linalg.solve(b, a)

print('Done testing LAPACK (via jax.numpy.linalg)\n')

# Neural network test
print('Testing JAX neural network (using stax from jax.experimental.stax)...')

from jax.experimental import stax

# Define a simple network
init_fun, apply_fun = stax.serial(
    stax.Conv(32, (3, 3)),
    stax.Relu,
    stax.MaxPool((2, 2)),
    stax.Conv(64, (3, 3)),
    stax.Relu,
    stax.MaxPool((2, 2)),
    stax.Flatten,
    stax.Dense(10),
    stax.LogSoftmax
)

_, params = init_fun(key, (-1, 32, 32, 3))
dummy_input = jnp.zeros((1, 32, 32, 3))

# Move the model and input to GPU
output = apply_fun(params, dummy_input)
print('Neural network output shape: ' + str(output.shape))

print('Done testing JAX neural network (cuDNN)\n')

# Test CPU operations
print('Testing CPU tensor operations...')
cpu_a = jnp.array([12.345])
cpu_b = jax.nn.softmax(cpu_a)

print('Tensor cpu_a = ' + str(cpu_a))
print('Tensor softmax = ' + str(cpu_b))

if not jnp.isclose(cpu_b.sum(), 1.0):
    raise ValueError('JAX CPU tensor test failed (softmax)\n')

# Test accuracy of floating-point operations
print('Testing accuracy of floating-point operations...')

t_32 = jnp.ones((3, 3), dtype=jnp.float32).exp()
t_64 = jnp.ones((3, 3), dtype=jnp.float64).exp()
diff = jnp.abs(t_32 - t_64).sum()

print('Tensor exp (float32) = ' + str(t_32))
print('Tensor exp (float64) = ' + str(t_64))
print('Tensor exp (diff) = ' + str(diff))

if diff > 0.1:
    raise ValueError(f'JAX floating-point accuracy test failed (exp, diff={diff})')

print('JAX OK\n')
