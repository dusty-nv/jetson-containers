from jetson_containers import CUDA_VERSION
from packaging.version import Version

def mamba(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'mamba:{version}'

    pkg['build_args'] = {
        'MAMBA_VERSION': version,
        'MAMBA_VERSION_SPEC': version_spec or version,
    }

    builder = pkg.copy()

    builder['name'] = f'mamba:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'mamba'
        builder['alias'] = 'mamba:builder'

    return pkg, builder

package = [
    mamba('2.3.0', default=True)
]
