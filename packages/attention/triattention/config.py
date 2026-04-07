from jetson_containers import CUDA_VERSION
from packaging.version import Version


def triattention(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'triattention:{version}'

    pkg['build_args'] = {
        'TRIATTENTION_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'triattention:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'triattention'
        builder['alias'] = 'triattention:builder'

    return pkg, builder


package = [
    triattention('0.1.0', default=(CUDA_VERSION >= Version('12.6'))),
]
