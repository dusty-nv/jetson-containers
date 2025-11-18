# this is in a different file from config.py so other packages can import it
from jetson_containers import L4T_VERSION, CUDA_VERSION
from packaging.version import Version

import os

if 'TENSORFLOW_VERSION' in os.environ and len(os.environ['TENSORFLOW_VERSION']) > 0:
    TENSORFLOW_VERSION = Version(os.environ['TENSORFLOW_VERSION'])
else:
    if L4T_VERSION.major >= 38:
        if CUDA_VERSION >= Version('13.0'):
            TENSORFLOW_VERSION = Version('2.21.0')
    if L4T_VERSION.major >= 36:
        if CUDA_VERSION >= Version('13.0'):
            TENSORFLOW_VERSION = Version('2.21.0')
        elif CUDA_VERSION >= Version('12.9'):
            TENSORFLOW_VERSION = Version('2.20.0')
        elif Version('12.4') <= CUDA_VERSION < Version('12.9'):
            TENSORFLOW_VERSION = Version('2.20.0')
        else:
            TENSORFLOW_VERSION = Version('2.16.1')
    elif L4T_VERSION.major >= 34:
        TENSORFLOW_VERSION = Version('2.16.1')
    elif L4T_VERSION.major >= 32:
        TENSORFLOW_VERSION = Version('1.15.5')
