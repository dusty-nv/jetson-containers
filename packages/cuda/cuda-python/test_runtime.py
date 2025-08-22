#!/usr/bin/env python3
print('testing cuda-python (runtime API)...')
import cuda
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except Exception:
    from importlib_metadata import PackageNotFoundError, version as _pkg_version  # type: ignore

_dist_names = (
    'cuda-python', 'cuda_python',
    'cuda-core', 'cuda_core',
    'cuda-bindings', 'cuda_bindings',
)

_cuda_py_version = None
for _name in _dist_names:
    try:
        _cuda_py_version = _pkg_version(_name)
        break
    except PackageNotFoundError:
        continue

print('cuda-python version:', _cuda_py_version)

try:
    from cuda.bindings import runtime as cudart  # cuda-python >= 13
except Exception:
    try:
        from cuda import runtime as cudart    # cuda-python ~ 12.4+
    except Exception:
        from cuda import cudart as cudart     # legacy name
from utils import checkCudaErrors

print('cuda driver version:', checkCudaErrors(cudart.cudaDriverGetVersion()))
print('cuda runtime version:', checkCudaErrors(cudart.cudaRuntimeGetVersion()))
print('cuda device count:', checkCudaErrors(cudart.cudaGetDeviceCount()))

print(checkCudaErrors(cudart.cudaGetDeviceProperties(0)))

print('cuda-python (runtime API) OK\n')
