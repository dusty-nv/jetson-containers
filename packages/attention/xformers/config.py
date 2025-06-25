from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def xformers(version, requires=None, default=True):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'xformers:{version}'

    pkg['build_args'] = {
        'XFORMERS_VERSION': version,
        'IS_SBSA': IS_SBSA,
    }

    builder = pkg.copy()

    builder['name'] = f'xformers:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'xformers'
        builder['alias'] = 'xformers:builder'

    return pkg, builder

package = [
    xformers('0.0.26', requires='<=cu122'),
    xformers('0.0.29', requires='<cu126'), # support pytorch 2.5.1
    xformers('0.0.30', default=False), # support pytorch 2.6.0
    xformers('0.0.33', default=(CUDA_VERSION >= Version('12.6'))), # Support Blackwell and pytorch 2.7.0
]
