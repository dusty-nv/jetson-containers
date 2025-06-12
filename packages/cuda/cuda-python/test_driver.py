#!/usr/bin/env python3
print('testing cuda-python (driver API)...')
import cuda
print('cuda-python version:', cuda.__version__)

import os
import math
import ctypes
import numpy as np

from cuda import cuda
from utils import checkCudaErrors, KernelHelper

checkCudaErrors(cuda.cuInit(0))

print('cuda driver version:', checkCudaErrors(cuda.cuDriverGetVersion()))
print('cuda device count:', checkCudaErrors(cuda.cuDeviceGetCount()))

cuDevice = cuda.CUdevice(0)
cuContext = checkCudaErrors(cuda.cuCtxCreate(0, cuDevice))

print('cuda device name:', checkCudaErrors(cuda.cuDeviceGetName(512, cuDevice)))

uvaSupported = checkCudaErrors(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cuDevice))

if not uvaSupported:
    raise RuntimeError("Accessing pageable memory directly requires UVA")
    
vectorAddDrv = '''\
/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

// Device code
extern "C" __global__ void VecAdd_kernel(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}
'''

kernelHelper = KernelHelper(vectorAddDrv, int(cuDevice))
_VecAdd_kernel = kernelHelper.getFunction(b'VecAdd_kernel')

N = 50000
size = N * np.dtype(np.float32).itemsize
    
# Allocate input vectors h_A and h_B in host memory
h_A = np.random.rand(size).astype(dtype=np.float32)
h_B = np.random.rand(size).astype(dtype=np.float32)
h_C = np.random.rand(size).astype(dtype=np.float32)

# Allocate vectors in device memory
d_A = checkCudaErrors(cuda.cuMemAlloc(size))
d_B = checkCudaErrors(cuda.cuMemAlloc(size))
d_C = checkCudaErrors(cuda.cuMemAlloc(size))

# Copy vectors from host memory to device memory
checkCudaErrors(cuda.cuMemcpyHtoD(d_A, h_A, size))
checkCudaErrors(cuda.cuMemcpyHtoD(d_B, h_B, size))

if True:
    # Grid/Block configuration
    threadsPerBlock = 256
    blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock

    kernelArgs = ((d_A, d_B, d_C, N),
                  (None, None, None, ctypes.c_int))

    # Launch the CUDA kernel
    checkCudaErrors(cuda.cuLaunchKernel(_VecAdd_kernel,
                                        blocksPerGrid, 1, 1,
                                        threadsPerBlock, 1, 1,
                                        0, cuda.CUstream(0),
                                        kernelArgs, 0))
else:
    pass

# Copy result from device memory to host memory
# h_C contains the result in host memory
checkCudaErrors(cuda.cuMemcpyDtoH(h_C, d_C, size))

for i in range(N):
    sum_all = h_A[i] + h_B[i]
    if math.fabs(h_C[i] - sum_all) > 1e-7:
        break

# Free device memory
checkCudaErrors(cuda.cuMemFree(d_A))
checkCudaErrors(cuda.cuMemFree(d_B))
checkCudaErrors(cuda.cuMemFree(d_C))

checkCudaErrors(cuda.cuCtxDestroy(cuContext))

# Check that kernel results were correct
print("{}".format("Result = PASS" if i+1 == N else "Result = FAIL"))

if i+1 != N:
    raise RuntimeError("VecAdd_kernel computed invalid results")
        
print('cuda-python (driver API) OK\n')
