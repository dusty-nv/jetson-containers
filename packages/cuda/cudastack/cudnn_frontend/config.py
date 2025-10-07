from jetson_containers import CUDA_VERSION, IS_SBSA, CUDA_ARCHITECTURES
from packaging.version import Version

def cudnn_frontend(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'cudnn_frontend:{version}'

    pkg['build_args'] = {
        'CUDNN_FRONTEND_VERSION': version,
        'CUDNN_FRONTEND_VERSION_SPEC': version_spec if version_spec else version,
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x / 10:.1f}' for x in CUDA_ARCHITECTURES]),
        'IS_SBSA': IS_SBSA,
    }

    builder = pkg.copy()

    builder['name'] = f'cudnn_frontend:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'cudnn_frontend'
        builder['alias'] = 'cudnn_frontend:builder'

    return pkg, builder


package = [
    cudnn_frontend('1.14.1', '1.14.1', default=(CUDA_VERSION >= Version('12.6'))),
]
