from jetson_containers import update_dependencies
from packaging.version import Version
from ..pytorch.version import PYTORCH_VERSION

def torchcodec(version, pytorch=None, requires=None):
    pkg = package.copy()

    pkg['name'] = f"torchcodec:{version.split('-')[0]}"  # remove any -rc* suffix

    if pytorch:
        pkg['depends'] = update_dependencies(pkg['depends'], f"pytorch:{pytorch}")
    else:
        pytorch = PYTORCH_VERSION

    if requires:
        pkg['requires'] = requires

    if not isinstance(pytorch, Version):
        pytorch = Version(pytorch)

    if len(version.split('.')) < 3:
        version = version + '.0'

    pkg['build_args'] = {
        'TORCHCODEC_VERSION': version,
    }

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if pytorch == PYTORCH_VERSION:
        pkg['alias'] = 'torchcodec'
        builder['alias'] = 'torchcodec:builder'

    return pkg, builder

package = [
    # JetPack 5/6/7
    torchcodec('0.6.0', pytorch='2.8', requires='>=36'),
    torchcodec('0.7.0', pytorch='2.9', requires='>=36'),
    torchcodec('0.8.0', pytorch='2.9', requires='>=36'),
    torchcodec('0.8.0', pytorch='2.10', requires='>=36'),
]
