#!/usr/bin/env python3
import torch
import numpy as np


class cudaArrayInterface():
    """
    Exposes __cuda_array_interface__ - typically used as a temporary view into a larger buffer
    https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    """
    def __init__(self, data, shape, dtype=np.float32):
        if dtype == np.float32:
            typestr = 'f4'
        elif dtype == np.float64:
            typestr = 'f8'
        elif dtype == np.float16:
            typestr = 'f2'
        else:
            raise RuntimeError(f"unsupported dtype:  {dtype}")
            
        self.__cuda_array_interface__ = {
            'data': (data, False),  # R/W
            'shape': shape,
            'typestr': typestr,
            'version': 3,
        }  
        

torch_dtype_dict = {
    'bool'       : torch.bool,
    'uint8'      : torch.uint8,
    'int8'       : torch.int8,
    'int16'      : torch.int16,
    'int32'      : torch.int32,
    'int64'      : torch.int64,
    'float16'    : torch.float16,
    'float32'    : torch.float32,
    'float64'    : torch.float64,
    'complex64'  : torch.complex64,
    'complex128' : torch.complex128
}

def torch_dtype(dtype):
    """
    Convert numpy.dtype or str to torch.dtype
    """
    return torch_dtype_dict[str(dtype)]