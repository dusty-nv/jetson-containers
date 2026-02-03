from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def mooncake(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if not version_spec:
        version_spec = version

    pkg['name'] = f'mooncake:{version}'

    pkg['build_args'] = {
        'MOONCAKE_VERSION': version,
        'MOONCAKE_VERSION_SPEC': version_spec,
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()

    builder['name'] = f'mooncake:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'mooncake'
        builder['alias'] = 'mooncake:builder'

    return pkg, builder

package = [
    mooncake('0.3.8.post1', '0.3.8.post1', default=True),
]
