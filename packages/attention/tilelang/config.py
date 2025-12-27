from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def tilelang(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'tilelang:{version}'

    pkg['build_args'] = {
        'TILELANG_VERSION': version,
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()

    builder['name'] = f'tilelang:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'tilelang'
        builder['alias'] = 'tilelang:builder'

    return pkg, builder

package = [
    tilelang('0.1.8', default=(CUDA_VERSION >= Version('12.6'))),
]

