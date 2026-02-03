from jetson_containers import CUDA_VERSION
from packaging.version import Version

def librealsense(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'librealsense:{version}'

    pkg['build_args'] = {
        'LIBREALSENSE_VERSION': version,
        'LIBREALSENSE_VERSION_SPEC': version_spec or version,
    }

    builder = pkg.copy()

    builder['name'] = f'librealsense:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'librealsense'
        builder['alias'] = 'librealsense:builder'

    return pkg, builder

package = [
    librealsense('2.54.2', default=False),
    librealsense('2.50.0', default=False),
    librealsense('2.57.6', default=True)
]
