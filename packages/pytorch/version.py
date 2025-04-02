# this is in a different file from config.py so other packages can import it
from jetson_containers import L4T_VERSION, CUDA_VERSION, SYSTEM_ARM
from packaging.version import Version

import os

if 'PYTORCH_VERSION' in os.environ and len(os.environ['PYTORCH_VERSION']) > 0:
    PYTORCH_VERSION = Version(os.environ['PYTORCH_VERSION'])
elif SYSTEM_ARM:
    if L4T_VERSION.major >= 36:
        if CUDA_VERSION >= Version('12.8'):   # JetPack 6.1 (CUDA 12.6)
            PYTORCH_VERSION = Version('2.6')
        elif CUDA_VERSION == Version('12.6'):   # JetPack 6.1 (CUDA 12.6)
            PYTORCH_VERSION = Version('2.6')
        elif CUDA_VERSION >= Version('12.4'): # JetPack 6.0 (CUDA 12.4)
            PYTORCH_VERSION = Version('2.4')
        else:
            PYTORCH_VERSION = Version('2.2')  # JetPack 6.0 (CUDA 12.2)
    elif L4T_VERSION.major >= 34:
        PYTORCH_VERSION = Version('2.2')      # JetPack 5.1 (CUDA 11.4)
    elif L4T_VERSION.major >= 32:
        PYTORCH_VERSION = Version('1.10')     # JetPack 4.6 (CUDA 10.2)
else:
    PYTORCH_VERSION = Version('2.6')          # pytorch nightly (CUDA 12.8)