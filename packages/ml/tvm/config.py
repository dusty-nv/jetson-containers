
from jetson_containers import CUDA_ARCHITECTURES

def tvm(commit='e3efec216f2d46033b69a51103cee174876cde18', version=None, default=True):
    pkg = package.copy()

    pkg['name'] = 'tvm'
    if default:
        pkg['alias'] = 'tvm'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]),
        'TVM_COMMIT': commit,
    }

    if version:
        pkg['build_args']['TVM_VERSION'] = version

    builder = pkg.copy()
    builder['name'] = 'tvm:builder'
    builder['build_args'] = {**pkg['build_args'], 'FORCE_BUILD': 'on'}

    return pkg, builder


package = [
    tvm(version=0.22, default=True)
]
