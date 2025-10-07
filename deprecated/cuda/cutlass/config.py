from jetson_containers import CUDA_VERSION
from packaging.version import Version

def cutlass(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'cutlass:{version}'

    pkg['build_args'] = {
        'CUTLASS_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'cutlass:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'cutlass'
        builder['alias'] = 'cutlass:builder'

    return pkg, builder

package = [
    cutlass('3.9.2', default=(CUDA_VERSION < Version('13.0'))),
    cutlass('4.2.0', default=(CUDA_VERSION >= Version('13.0'))),
]
