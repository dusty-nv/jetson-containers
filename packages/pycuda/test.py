#!/usr/bin/env python3
print('testing PyCUDA...')

import pycuda
import pycuda.driver as cuda
import pycuda.autoinit

from pycuda.compiler import SourceModule

import numpy as np


# print device info
print('PyCUDA version:      ' + str(pycuda.VERSION_TEXT))
print('CUDA build version:  ' + str(cuda.get_version()))
print('CUDA driver version: ' + str(cuda.get_driver_version()))

dev = cuda.Device(0)

print('CUDA device name:    ' + str(dev.name()))
print('CUDA device memory:  ' + str((int)(dev.total_memory()/1048576)) + ' MB')
print('CUDA device compute: ' + str(dev.compute_capability()))

# allocate memory
print('allocating device memory...')

a_cpu = np.full((4,8), 1.0, np.float32)
b_cpu = np.full(a_cpu.shape, 2.0, np.float32)
c_cpu = np.empty_like(a_cpu)

a_gpu = cuda.mem_alloc(a_cpu.nbytes)
b_gpu = cuda.mem_alloc(b_cpu.nbytes)
c_gpu = cuda.mem_alloc(b_cpu.nbytes)

cuda.memcpy_htod(a_gpu, a_cpu)
cuda.memcpy_htod(b_gpu, b_cpu)

# test cuda kernel
print('building CUDA kernel...')

module = SourceModule("""
    __global__ void cuda_add( float* a, float* b, float* c )
    {
        int idx = threadIdx.x + threadIdx.y * blockDim.x;
        c[idx] = a[idx] + b[idx];
    }
    """)
    
func = module.get_function('cuda_add')

print('running CUDA kernel...')
func(a_gpu, b_gpu, c_gpu, block=(a_cpu.shape[0], b_cpu.shape[1], 1))
cuda.memcpy_dtoh(c_cpu, c_gpu)
print('CUDA kernel results:')
print(c_cpu)  # should be '3.0'

print('PyCUDA OK\n')
