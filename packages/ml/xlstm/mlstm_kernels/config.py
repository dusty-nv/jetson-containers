from jetson_containers import CUDA_VERSION
from packaging.version import Version

def mlstm_kernels(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'mlstm_kernels:{version}'

    pkg['build_args'] = {
        'MLSTM_KERNELS_VERSION': version,
        'MLSTM_KERNELS_VERSION_SPEC': version_spec if version_spec else version,
    }

    builder = pkg.copy()

    builder['name'] = f'mlstm_kernels:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'off'}}

    if default:
        pkg['alias'] = 'mlstm_kernels'
        builder['alias'] = 'mlstm_kernels:builder'

    return pkg, builder

package = [
    mlstm_kernels('3.0.0', version_spec='3.0.0', default=True)
]
