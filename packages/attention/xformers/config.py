from jetson_containers import CUDA_VERSION
from packaging.version import Version
from ..ml.pytorch.version import PYTORCH_VERSION
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
    xformers('0.0.35', default=(CUDA_VERSION >= Version('12.6'))),
]
