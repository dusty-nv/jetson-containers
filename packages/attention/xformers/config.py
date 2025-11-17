from jetson_containers import CUDA_VERSION
from packaging.version import Version
from ..pytorch.version import PYTORCH_VERSION
from jetson_containers import update_dependencies


def xformers(version, requires=None, pytorch=None, default=True):
    pkg = package.copy()

    if pytorch:
        pkg['depends'] = update_dependencies(pkg['depends'], f"pytorch:{pytorch}")
    else:
        pytorch = PYTORCH_VERSION

    if requires:
        pkg['requires'] = requires

    if not isinstance(pytorch, Version):
        pytorch = Version(pytorch)

    pkg['name'] = f'xformers:{version}'

    pkg['build_args'] = {
        'XFORMERS_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'xformers:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if pytorch == PYTORCH_VERSION and default:
        pkg['alias'] = 'xformers'
        builder['alias'] = 'xformers:builder'

    return pkg, builder


package = [
    xformers('0.0.26', pytorch='2.1', requires='<=cu122'),
    xformers('0.0.29', pytorch='2.4', requires='<cu126'),  # support pytorch 2.5.1
    xformers('0.0.30', pytorch='2.7', default=(CUDA_VERSION < Version('12.6'))),
    # support pytorch 2.6.0
    xformers('0.0.32.post2', pytorch='2.8', default=(CUDA_VERSION >= Version('12.6'))),
    # support pytorch 2.8.0
    xformers('0.0.34', default=(CUDA_VERSION >= Version('12.6'))),
    # Support Blackwell and pytorch 2.9.0
]
