from jetson_containers import CUDA_VERSION
from packaging.version import Version

def pltsm(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'pltsm:{version}'

    pkg['build_args'] = {
        'PLSTM_VERSION': version,
        'PLSTM_VERSION_SPEC': version_spec if version_spec else version,
    }

    builder = pkg.copy()

    builder['name'] = f'pltsm:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'pltsm'
        builder['alias'] = 'pltsm:builder'

    return pkg, builder

package = [
    pltsm('0.1.0', version_spec='0.1.0', default=True)
]
