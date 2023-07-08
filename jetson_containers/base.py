#!/usr/bin/env python3
from .l4t_version import L4T_VERSION

def get_l4t_base():
    """
    Returns the l4t-base or l4t-jetpack container to use
    """
    if L4T_VERSION.major >= 5:
        return f"nvcr.io/nvidia/l4t-jetpack:r{L4T_VERSION}"
    else:
        return f"nvcr.io/nvidia/l4t-base:r{L4T_VERSION}"
