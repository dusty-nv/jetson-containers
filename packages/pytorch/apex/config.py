
from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def apex(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'apex:{version}'

    pkg['build_args'] = {
        'APEX_VERSION': version,
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()

    builder['name'] = f'apex:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'apex'
        builder['alias'] = 'apex:builder'

    return pkg, builder

package = [
    apex('0.1', default=(CUDA_VERSION >= Version('12.6'))),
]

