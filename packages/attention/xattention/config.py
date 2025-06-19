from jetson_containers import CUDA_VERSION
from packaging.version import Version

def xattention(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'xattention:{version}'

    pkg['build_args'] = {
        'XATTENTION_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'xattention:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'xattention'
        builder['alias'] = 'xattention:builder'

    return pkg, builder

package = [
    xattention('0.0.1', default=(CUDA_VERSION >= Version('12.6'))),
]
