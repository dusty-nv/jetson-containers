# version.py
# This file defines the TensorFlow version and maintains the URLs and wheel filenames.
# Other packages can import variables from this file.

from jetson_containers import L4T_VERSION, CUDA_VERSION
from packaging.version import Version

import os

# Determine the TensorFlow version
if 'TENSORFLOW_VERSION' in os.environ and len(os.environ['TENSORFLOW_VERSION']) > 0:
    TENSORFLOW_VERSION = Version(os.environ['TENSORFLOW_VERSION'])
else:
    if L4T_VERSION.major >= 36:
        if CUDA_VERSION >= Version('12.4'):
            TENSORFLOW_VERSION = Version('2.18.0')
        else:
            TENSORFLOW_VERSION = Version('2.16.1')
    elif L4T_VERSION.major == 35:
        TENSORFLOW_VERSION = Version('2.11.0')
    elif L4T_VERSION.major == 34:
        TENSORFLOW_VERSION = Version('2.8.0')
    elif L4T_VERSION.major == 32:
        TENSORFLOW_VERSION = Version('2.7.0')
    else:
        # Default to the latest known version if L4T_VERSION is unrecognized
        TENSORFLOW_VERSION = Version('2.16.1')

# Define the URLs and wheel filenames for TensorFlow 1 and 2
if L4T_VERSION.major >= 36:    # JetPack 6.0
    TENSORFLOW1_URL = None
    TENSORFLOW1_WHL = None
    if TENSORFLOW_VERSION == Version('2.16.1'):
        TENSORFLOW2_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v60/tensorflow/tensorflow-2.16.1+nv24.07-cp310-cp310-linux_aarch64.whl'
        TENSORFLOW2_WHL = 'tensorflow-2.16.1+nv24.07-cp310-cp310-linux_aarch64.whl'
    elif TENSORFLOW_VERSION == Version('2.13.0'):
        TENSORFLOW2_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v60/tensorflow/tensorflow-2.13.0+nv23.07-cp310-cp310-linux_aarch64.whl'
        TENSORFLOW2_WHL = 'tensorflow-2.13.0+nv23.07-cp310-cp310-linux_aarch64.whl'
    else:
        TENSORFLOW2_URL = None
        TENSORFLOW2_WHL = None
elif L4T_VERSION.major == 35:  # JetPack 5.1.x
    TENSORFLOW1_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v51/tensorflow/tensorflow-1.15.5+nv23.03-cp38-cp38-linux_aarch64.whl'
    TENSORFLOW1_WHL = 'tensorflow-1.15.5+nv23.03-cp38-cp38-linux_aarch64.whl'
    if TENSORFLOW_VERSION == Version('2.11.0'):
        TENSORFLOW2_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v51/tensorflow/tensorflow-2.11.0+nv23.03-cp38-cp38-linux_aarch64.whl'
        TENSORFLOW2_WHL = 'tensorflow-2.11.0+nv23.03-cp38-cp38-linux_aarch64.whl'
    else:
        TENSORFLOW2_URL = None
        TENSORFLOW2_WHL = None
elif L4T_VERSION.major == 34:  # JetPack 5.0 / 5.0.1
    TENSORFLOW1_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/tensorflow-1.15.5+nv22.4-cp38-cp38-linux_aarch64.whl'
    TENSORFLOW1_WHL = 'tensorflow-1.15.5+nv22.4-cp38-cp38-linux_aarch64.whl'
    if TENSORFLOW_VERSION == Version('2.8.0'):
        TENSORFLOW2_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/tensorflow-2.8.0+nv22.4-cp38-cp38-linux_aarch64.whl'
        TENSORFLOW2_WHL = 'tensorflow-2.8.0+nv22.4-cp38-cp38-linux_aarch64.whl'
    else:
        TENSORFLOW2_URL = None
        TENSORFLOW2_WHL = None
elif L4T_VERSION.major == 32:  # JetPack 4.x
    TENSORFLOW1_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-1.15.5+nv22.1-cp36-cp36m-linux_aarch64.whl'
    TENSORFLOW1_WHL = 'tensorflow-1.15.5+nv22.1-cp36-cp36m-linux_aarch64.whl'
    if TENSORFLOW_VERSION == Version('2.7.0'):
        TENSORFLOW2_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl'
        TENSORFLOW2_WHL = 'tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl'
    else:
        TENSORFLOW2_URL = None
        TENSORFLOW2_WHL = None
else:
    # If L4T_VERSION is unrecognized, set URLs and wheel filenames to None
    TENSORFLOW1_URL = None
    TENSORFLOW1_WHL = None
    TENSORFLOW2_URL = None
    TENSORFLOW2_WHL = None
