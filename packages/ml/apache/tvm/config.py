
from jetson_containers import CUDA_ARCHITECTURES

def tvm(version=None, default=True):
    pkg = package.copy()

    pkg['name'] = 'tvm'
    if default:
        pkg['alias'] = 'tvm'

    pkg['build_args'] = {
        'TVM_VERSION': version,
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]),
    }

    builder = pkg.copy()
    builder['name'] = 'tvm:builder'
    builder['build_args'] = {**pkg['build_args'], 'FORCE_BUILD': 'off'}

    return pkg, builder


package = [
    tvm(version="0.23.0", default=True)
]
