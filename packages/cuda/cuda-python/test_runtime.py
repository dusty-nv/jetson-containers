#!/usr/bin/env python3
print('testing cuda-python (runtime API)...')
import cuda
print('cuda-python version:', cuda.__version__)

from cuda import cudart
from utils import checkCudaErrors

print('cuda driver version:', checkCudaErrors(cudart.cudaDriverGetVersion()))
print('cuda runtime version:', checkCudaErrors(cudart.cudaRuntimeGetVersion()))
print('cuda device count:', checkCudaErrors(cudart.cudaGetDeviceCount()))

print(checkCudaErrors(cudart.cudaGetDeviceProperties(0)))

print('cuda-python (runtime API) OK\n')
