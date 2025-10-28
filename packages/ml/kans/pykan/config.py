from jetson_containers import CUDA_VERSION
from packaging.version import Version

def pykan(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'pykan:{version}'

    pkg['build_args'] = {
        'PYKAN_VERSION': version,
        'PYKAN_VERSION_SPEC': version_spec if version_spec else version,
    }

    builder = pkg.copy()

    builder['name'] = f'pykan:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'pykan'
        builder['alias'] = 'pykan:builder'

    return pkg, builder

package = [
    pykan('0.3.9', version_spec='0.3.9', default=True)
]
