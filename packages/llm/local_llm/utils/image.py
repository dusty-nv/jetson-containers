#!/usr/bin/env python3
import io
import PIL
import logging
import torch
import numpy as np

from jetson_utils import cudaImage, cudaFromNumpy


ImageTypes = (PIL.Image.Image, np.ndarray, torch.Tensor, cudaImage)
ImageExtensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')


def is_image(image):
    """
    Returns true if the object is a PIL.Image, np.ndarray, torch.Tensor, or jetson_utils.cudaImage
    """
    return isinstance(image, ImageTypes)
    
 
def image_size(image):
    """
    Returns the dimensions of the image as a tuple (height, width, channels)
    """
    if isinstance(image, (cudaImage, np.ndarray, torch.Tensor)):
        return image.shape
    elif isinstance(image, PIL.Image.Image):
        return image.size
    else:
        raise TypeError(f"expected an image of type {ImageTypes} (was {type(image)})")
        
    
def load_image(path):
    """
    Load an image from a local path or URL
    TODO have this use jetson_utils instead
    """
    if path.startswith('http') or path.startswith('https'):
        logging.debug(f'-- downloading {path}')
        response = requests.get(path)
        image = PIL.Image.open(io.BytesIO(response.content)).convert('RGB')
    else:
        logging.debug(f'-- loading {path}')
        image = PIL.Image.open(path).convert('RGB')
        
    return image


def cuda_image(image):
    """
    Convert an image from PIL.Image, np.ndarray, torch.Tensor, or __gpu_array_interface__
    to a jetson_utils.cudaImage on the GPU (without using memory copies when possible)
    
    TODO implement __gpu_array_interface__
    TODO torch image formats https://github.com/dusty-nv/jetson-utils/blob/f0bff5c502f9ac6b10aa2912f1324797df94bc2d/python/examples/cuda-from-pytorch.py#L47
    """
    if not is_image(image):
        raise TypeError(f"expected an image of type {ImageTypes} (was {type(image)})")
        
    if isinstance(image, cudaImage):
        return image
        
    if isinstance(image, PIL.Image.Image):
        image = np.asarray(image)  # no copy
        
    if isinstance(image, np.ndarray):
        return cudaFromNumpy(image)
        
    if isinstance(image, torch.Tensor):
        input = input.to(memory_format=torch.channels_last)   # or tensor.permute(0, 3, 2, 1)
        
        return cudaImage(
            ptr=input.data_ptr(), 
            width=input.shape[-1], 
            height=input.shape[-2], 
            format=torch_image_format(input)
        )
        
 
def torch_image(image):
    """
    Convert the image to a type that is compatible with PyTorch (torch.Tensor, ndarray, PIL.Image)
    """
    if isinstance(image, cudaImage):
        return torch.as_tensor(image, device='cuda')
    elif is_image(image):
        return image 
    raise TypeError(f"expected an image of type {ImageTypes} (was {type(image)})")
        
        
def torch_image_format(tensor):
    """
    Determine the cudaImage format string (eg 'rgb32f', 'rgba32f', ect) from a PyTorch tensor.
    Only float and uint8 tensors are supported because those datatypes are supported by cudaImage.
    """
    if tensor.dtype != torch.float32 and tensor.dtype != torch.uint8:
        raise ValueError(f"PyTorch tensor datatype should be torch.float32 or torch.uint8 (was {tensor.dtype})")
        
    if len(tensor.shape)>= 4:     # NCHW layout
        channels = tensor.shape[1]
    elif len(tensor.shape) == 3:   # CHW layout
        channels = tensor.shape[0]
    elif len(tensor.shape) == 2:   # HW layout
        channels = 1
    else:
        raise ValueError(f"PyTorch tensor should have at least 2 image dimensions (has {tensor.shape.length})")
        
    if channels == 1:   return 'gray32f' if tensor.dtype == torch.float32 else 'gray8'
    elif channels == 3: return 'rgb32f'  if tensor.dtype == torch.float32 else 'rgb8'
    elif channels == 4: return 'rgba32f' if tensor.dtype == torch.float32 else 'rgba8'
    
    raise ValueError(f"PyTorch tensor should have 1, 3, or 4 image channels (has {channels})")
    