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
    
def convert_dtype(dtype, to='np'):
    """
    Convert a string, numpy type, or torch.dtype to either numpy or PyTorch
    """
    if to == 'pt':
        if isinstance(dtype, torch.dtype):
            return dtype
        else:
            return torch_dtype(dtype)
    elif to == 'np':
        if isinstance(dtype, type):
            return dtype
        elif isinstance(dtype, torch.dtype):
            return np.dtype(str(dtype).split('.')[-1]) # remove the torch.* prefix
        else:
            return np.dtype(dtype)
            
    raise TypeError(f"expected dtype as a string, type, or torch.dtype (was {type(dtype)}) and with to='np' or to='pt'")
    
def convert_tensor(tensor, return_tensors='pt', device=None, dtype=None, **kwargs):
    """
    Convert tensors between numpy/torch/ect
    """
    if tensor is None:
        return None
        
    if isinstance(tensor, np.ndarray):
        if return_tensors == 'np':
            return tensor
        elif return_tensors == 'pt':
            return torch.from_numpy(tensor).to(device=device, dtype=dtype, **kwargs)
    elif isinstance(tensor, torch.Tensor):
        if return_tensors == 'np':
            if dtype:
                tensor = tensor.to(dtype=dtype)
            return tensor.detach().cpu().numpy()
        elif return_tensors == 'pt':
            if device or dtype:
                return tensor.to(device=device, dtype=dtype, **kwargs)
            
    raise ValueError(f"unsupported tensor input/output type (in={type(tensor)} out={return_tensors})")