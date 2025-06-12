#!/usr/bin/env python3
print('testing PyTorch...')

import torch

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
print(f'cuDNN version:   {torch.backends.cudnn.version()}\n')

print(torch.__config__.show())

# fail if CUDA isn't available
assert(torch.cuda.is_available())

print(f'\nPyTorch {torch.__version__}\n')

try:
    print(f'  * CUDA device     {torch.cuda.get_device_name()}')
    print(f'  * CUDA version    {torch.version.cuda}')
    print(f'  * CUDA cuDNN      {torch.backends.cudnn.version()}')
    print(f'  * CUDA BLAS       {torch.backends.cuda.preferred_blas_library()}')
    print(f'  * CUDA linalg     {torch.backends.cuda.preferred_blas_library()}')
    print(f'  * CUDA flash_attn {torch.backends.cuda.is_flash_attention_available()}')
    print(f'  * CUDA flash_sdp  {torch.backends.cuda.flash_sdp_enabled()}')
    print(f'  * CUDA cudnn_sdp  {torch.backends.cuda.cudnn_sdp_enabled()}')
    print(f'  * CUDA math_sdp   {torch.backends.cuda.math_sdp_enabled()}')
    print(f'  * CUDA mem_efficient_sdp_enabled    {torch.backends.cuda.mem_efficient_sdp_enabled()}')
    print(f'  * CUDA fp16_bf16_reduction_math_sdp {torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()}')
except Exception as error:
    print(f'Exception trying to read PyTorch {torch.__version__} CUDA versions (this may be expected on older versions of PyTorch)\n{error}')

print(f'\ntorch.distributed: {torch.distributed.is_available()}')
# NCCL backend check
try:
    import torch.distributed.nccl
    print("  * NCCL backend is present.")
except ImportError:
    print("  * NCCL backend is NOT present.")

# GLOO backend check
try:
    import torch.distributed.gloo
    print("  * GLOO backend is present.")
except ImportError:
    print("  * GLOO backend is NOT present.")

# MPI backend check
try:
    import torch.distributed.mpi
    print("  * MPI backend is present.\n")
except ImportError:
    print("  * MPI backend is NOT present.\n")

# check that version can be parsed
from packaging import version
from os import environ

print('PACKAGING_VERSION=' + str(version.parse(torch.__version__)))
print('TORCH_CUDA_ARCH_LIST=' + environ.get('TORCH_CUDA_ARCH_LIST', 'None') + '\n')

# quick cuda tensor test
a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))

b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))

c = a + b
print('Tensor c = ' + str(c))

# LAPACK test
print('testing LAPACK (OpenBLAS)...')

try:
    a = torch.randn(2, 3, 1, 4, 4)
    b = torch.randn(2, 3, 1, 4, 4)
    x, lu = torch.linalg.solve(b, a)
    print('done testing LAPACK (OpenBLAS)')
except Exception as e:
    print(f'❌LAPACK test failed: {e}')

# torch.nn test
print('testing torch.nn (cuDNN)...')

import torch.nn

model = torch.nn.Conv2d(3,3,3)
data = torch.zeros(1,3,10,10)
model = model.cuda()
data = data.cuda()
out = model(data)

#print(out)

print('done testing torch.nn (cuDNN)')

# CPU test (https://github.com/pytorch/pytorch/issues/47098)
print('testing CPU tensor vector operations...')

import torch.nn.functional as F
cpu_x = torch.tensor([12.345])
cpu_y = F.softmax(cpu_x)

print('Tensor cpu_x = ' + str(cpu_x))
print('Tensor softmax = ' + str(cpu_y))

if cpu_y != 1.0:
    raise ValueError('PyTorch CPU tensor vector test failed (softmax)\n')

# https://github.com/pytorch/pytorch/issues/61110
t_32 = torch.ones((3,3), dtype=torch.float32).exp()
t_64 = torch.ones((3,3), dtype=torch.float64).exp()
diff = (t_32 - t_64).abs().sum().item()

print('Tensor exp (float32) = ' + str(t_32))
print('Tensor exp (float64) = ' + str(t_64))
print('Tensor exp (diff) = ' + str(diff))

if diff > 0.1:
    raise ValueError(f'PyTorch CPU tensor vector test failed (exp, diff={diff})')

print('PyTorch OK\n')
