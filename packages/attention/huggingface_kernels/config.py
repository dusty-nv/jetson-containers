from jetson_containers import CUDA_VERSION, IS_SBSA, CUDA_ARCHITECTURES
from packaging.version import Version

def kernels(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'huggingface_kernels:{version}'

    pkg['build_args'] = {
        'KERNELS_VERSION': version,
        'IS_SBSA': IS_SBSA,
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x / 10:.1f}' for x in CUDA_ARCHITECTURES])
    }

    builder = pkg.copy()

    builder['name'] = f'huggingface_kernels:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'huggingface_kernels'
        builder['alias'] = 'huggingface_kernels:builder'

    return pkg, builder

package = [
    kernels('0.10.5', default=(CUDA_VERSION >= Version('12.6'))),
]

