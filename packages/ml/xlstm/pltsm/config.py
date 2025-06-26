from jetson_containers import CUDA_VERSION
from packaging.version import Version

def plstm(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'plstm:{version}'

    pkg['build_args'] = {
        'PLSTM_VERSION': version,
        'PLSTM_VERSION_SPEC': version_spec if version_spec else version,
    }

    builder = pkg.copy()

    builder['name'] = f'plstm:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'off'}}

    if default:
        pkg['alias'] = 'plstm'
        builder['alias'] = 'plstm:builder'

    return pkg, builder

package = [
    plstm('0.1.0', version_spec='0.1.0', default=True)
]
