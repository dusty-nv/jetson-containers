from jetson_containers import CUDA_VERSION
from packaging.version import Version

def xlstm(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'xlstm:{version}'

    pkg['build_args'] = {
        'XLSTM_VERSION': version,
        'XLSTM_VERSION_SPEC': version_spec if version_spec else version,
    }

    builder = pkg.copy()

    builder['name'] = f'xlstm:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'xlstm'
        builder['alias'] = 'xlstm:builder'

    return pkg, builder

package = [
    xlstm('2.0.5', version_spec='2.0.4', default=True)
]
