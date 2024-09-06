# this is in a different file from config.py so other packages can import it
from jetson_containers import L4T_VERSION, CUDA_VERSION
from packaging.version import Version

import os

if 'JAX_VERSION' in os.environ and len(os.environ['JAX_VERSION']) > 0:
    JAX_VERSION = Version(os.environ['JAX_VERSION'])
else:
    if L4T_VERSION.major >= 36:
        if CUDA_VERSION >= Version('12.4'):
            JAX_VERSION = Version('2.0')
        else:
            JAX_VERSION = Version('0.2')
    elif L4T_VERSION.major >= 34:
        JAX_VERSION = Version('0.2')
    elif L4T_VERSION.major >= 32:
        JAX_VERSION = Version('0.2')
