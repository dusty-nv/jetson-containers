# this is in a different file from config.py so other packages can import it
from jetson_containers import L4T_VERSION, CUDA_VERSION
from packaging.version import Version

import os

if 'PYTORCH_VERSION' in os.environ and len(os.environ['PYTORCH_VERSION']) > 0:
    PYTORCH_VERSION = Version(os.environ['PYTORCH_VERSION'])
else:
    if L4T_VERSION.major >= 36:
        if CUDA_VERSION >= Version('12.4'):
            PYTORCH_VERSION = Version('2.3')
        else:
            PYTORCH_VERSION = Version('2.2')
    elif L4T_VERSION.major >= 34:
        PYTORCH_VERSION = Version('2.2')
    elif L4T_VERSION.major >= 32:
        PYTORCH_VERSION = Version('1.10')
