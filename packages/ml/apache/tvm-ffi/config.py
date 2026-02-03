
from jetson_containers import CUDA_ARCHITECTURES

def tvm_ffi(version=None, default=True):
    pkg = package.copy()

    pkg['name'] = 'tvm_ffi'
    if default:
        pkg['alias'] = 'tvm_ffi'

    pkg['build_args'] = {
        'TVM_FFI_VERSION': version,
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]),
    }

    builder = pkg.copy()
    builder['name'] = 'tvm_ffi:builder'
    builder['build_args'] = {**pkg['build_args'], 'FORCE_BUILD': 'off'}

    return pkg, builder


package = [
    tvm_ffi(version="0.1.9", default=True)
]
