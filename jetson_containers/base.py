#!/usr/bin/env python3
from .l4t_version import L4T_VERSION
from packaging.version import Version

def get_l4t_base():
    """
    Returns the l4t-base or l4t-jetpack container to use
    """
    if L4T_VERSION.major >= 34:   # JetPack 5
        if L4T_VERSION >= Version('35.3.1'):
            return "nvcr.io/nvidia/l4t-jetpack:r35.3.1"
        else:
            return f"nvcr.io/nvidia/l4t-jetpack:r{L4T_VERSION}"
    else:
        if L4T_VERSION >= Version('32.7.1'):
            return "nvcr.io/nvidia/l4t-jetpack:r32.7.1"
        else:
            return f"nvcr.io/nvidia/l4t-base:r{L4T_VERSION}"
