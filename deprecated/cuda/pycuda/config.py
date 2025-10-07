from jetson_containers import CUDA_VERSION
from packaging.version import Version

def pycuda(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'pycuda:{version}'

    pkg['build_args'] = {
        'PYCUDA_VERSION': version
    }

    builder = pkg.copy()

    builder['name'] = f'pycuda:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'pycuda'
        builder['alias'] = 'pycuda:builder'

    return pkg, builder

package = [
    pycuda('2025.1.2', default=(CUDA_VERSION >= Version('12.6'))),
]
