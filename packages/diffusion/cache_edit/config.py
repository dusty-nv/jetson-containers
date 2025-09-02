from jetson_containers import CUDA_VERSION
from packaging.version import Version

def cache_dit(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'cache_dit:{version}'

    pkg['build_args'] = {
        'CACHE_DIT_VERSION': version,
        'CACHE_DIT_VERSION_SPEC': version_spec if version_spec else version,
    }

    builder = pkg.copy()

    builder['name'] = f'cache_dit:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'cache_dit'
        builder['alias'] = 'cache_dit:builder'

    return pkg, builder

package = [
    cache_dit('0.2.28', version_spec='0.2.28', default=True)
]
