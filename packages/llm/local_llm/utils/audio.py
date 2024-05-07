#!/usr/bin/env python3
import torch
import numpy as np

from .tensor import convert_dtype

def convert_audio(samples, dtype=np.int16):
    """
    Convert between audio datatypes like float<->int16 and apply sample re-scaling.
    If the samples are a raw bytes array, it's assumed that they are in int16 format.
    Supports audio samples as byte buffer, numpy ndarray, and torch.Tensor.  Converted
    byte buffers will be returned as ndarray, otherwise the same object type as input.
    """
    if isinstance(samples, bytes):
        samples = np.frombuffer(samples, dtype=np.int16)
    elif not isinstance(samples, (np.ndarray, torch.Tensor)):
        raise TypeError(f"samples should either be bytes, np.ndarray, or torch.Tensor (was {type(samples)})")
        
    if samples.dtype == dtype:
        return samples

    #sample_width = np.dtype(str(dtype).split('.')[-1]).itemsize
    sample_width = np.dtype(convert_dtype(dtype, to='np')).itemsize
    max_value = float(int((2 ** (sample_width * 8)) / 2) - 1)  # 32767 for 16-bit
        
    if isinstance(samples, np.ndarray):
        if samples.dtype == np.float32 or samples.dtype == np.float64:  # float-to-int
            samples = samples * max_value
            samples = samples.clip(-max_value, max_value)
            samples = samples.astype(dtype)
        elif dtype == np.float32 or dtype == np.float64:  # int-to-float
            samples = samples.astype(dtype)
            samples = samples / max_value
        else:
            raise TypeError(f"unsupported audio sample dtype={samples.dtype}")
    elif isinstance(samples, torch.Tensor):
        if samples.dtype == torch.float32 or samples.dtype == torch.float64:
            samples = samples * max_value
            samples = samples.clip(-max_value, max_value).to(dtype=dtype)
        elif dtype == np.float32 or dtype == np.float64:
            samples = samples.to(dtype=dtype) / max_value
        else:
            raise TypeError(f"unsupported audio sample dtype={samples.dtype}")
            
    return samples


def audio_rms(samples):
    """
    Compute the average audio RMS (returns a float between 0 and 1)
    """
    return np.sqrt(np.mean(convert_audio(samples, dtype=np.float32)**2))


def audio_silent(samples, threshold=0.0):
    """
    Detect if the audio samples are silent or muted.
    
    If threshold < 0, false will be returned (silence detection disabled).
    If threshold > 0, the audio's average RMS will be compared to the threshold.
    If threshold = 0, it will check for any non-zero samples (faster than RMS)
    
    Returns true if audio levels are below threshold, otherwise false.
    """
    if threshold < 0:
        return False
        #raise ValueError("silence threshold should be >= 0")
        
    if threshold == 0:
        if isinstance(samples, bytes):
            samples = np.frombuffer(samples, dtype=np.int16)
        nonzero = np.count_nonzero(samples)
        return (nonzero == 0)
    else:       
        return audio_rms(samples) <= threshold
