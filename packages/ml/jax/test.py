#!/usr/bin/env python3
print('Testing JAX...')

import jax
import jax.numpy as jnp
from jax import random
import os

jax.print_environment_info()

# Print JAX version and CUDA device information
print('JAX version: ' + str(jax.__version__))
print('CUDA devices: ' + str(jax.devices()))
print('Default backend: ' + jax.default_backend())

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

a = jnp.asarray([[1.0, 2.0, 3.0], [4., 5., 6.]])
b = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5., 6.]])

print("Array 'a':")
print(a)
print("Array 'b':")
print(b)

print("where do the arrays live?")
print("Array 'a':", a.device, a.dtype, a.shape)
print("Array 'b':", b.device, b.dtype, b.shape)

print("Result of jnp.dot(a,b)")
c = jnp.dot(a,b)
print(c)
print("Array 'c':", c.device, c.dtype, c.shape)
print('JAX OK\n')
