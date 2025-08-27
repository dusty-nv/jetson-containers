from jetson_containers import CUDA_VERSION, IS_SBSA, CUDA_ARCHITECTURES
from packaging.version import Version

def cuda_cccl(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'cuda_cccl:{version}'

    pkg['build_args'] = {
        'CCCL_VERSION': version,
        'CCCL_VERSION_SPEC': version_spec if version_spec else version,
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x / 10:.1f}' for x in CUDA_ARCHITECTURES]),
        'IS_SBSA': IS_SBSA,
    }

    builder = pkg.copy()

    builder['name'] = f'cuda_cccl:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'cuda_cccl'
        builder['alias'] = 'cuda_cccl:builder'

    return pkg, builder


package = [
    cuda_cccl('3.1.0', '3.1.0', default=(CUDA_VERSION >= Version('12.6'))),
]
