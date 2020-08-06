
print('testing PyCUDA...')

import pycuda
import pycuda.driver as cuda
import pycuda.autoinit

print('PyCUDA version:      ' + str(pycuda.VERSION_TEXT))
print('CUDA build version:  ' + str(cuda.get_version()))
print('CUDA driver version: ' + str(cuda.get_driver_version()))

dev = cuda.Device(0)

print('CUDA device name:    ' + str(dev.name()))
print('CUDA device memory:  ' + str((int)(dev.total_memory()/1048576)) + ' MB')
print('CUDA device compute: ' + str(dev.compute_capability()))

print('PyCUDA OK\n')
