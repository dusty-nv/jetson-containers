from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def nixl(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if not version_spec:
        version_spec = version

    pkg['name'] = f'nixl:{version}'

    pkg['build_args'] = {
        'NIXL_VERSION': version,
        'NIXL_VERSION_SPEC': version_spec,
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()

    builder['name'] = f'nixl:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'nixl'
        builder['alias'] = 'nixl:builder'

    return pkg, builder

package = [
    nixl('0.3.0', '0.2.0', default=True),
]