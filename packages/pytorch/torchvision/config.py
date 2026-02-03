from jetson_containers import update_dependencies
from packaging.version import Version
from ..pytorch.version import PYTORCH_VERSION

def torchvision(version, pytorch=None, requires=None):
    pkg = package.copy()

    pkg['name'] = f"torchvision:{version.split('-')[0]}"  # remove any -rc* suffix

    if pytorch:
        pkg['depends'] = update_dependencies(pkg['depends'], f"pytorch:{pytorch}")
    else:
        pytorch = PYTORCH_VERSION

    if requires:
        pkg['requires'] = requires

    if len(version.split('.')) < 3:
        version = version + '.0'

    pkg['build_args'] = {
        'TORCHVISION_VERSION': version,
    }

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if not isinstance(pytorch, Version):
        pytorch = Version(pytorch)

    if pytorch == PYTORCH_VERSION:
        pkg['alias'] = 'torchvision'
        builder['alias'] = 'torchvision:builder'

    return pkg, builder


package = [
    # JetPack 5/6 and x86
    torchvision('0.15.1', pytorch='2.0', requires='==35.*'),
    torchvision('0.17.2', pytorch='2.2', requires='>=35'),
    torchvision('0.22.0', pytorch='2.7', requires='==36.*'),
    torchvision('0.24.1', pytorch='2.9.1', requires='>=36'),
    torchvision('0.25.0', pytorch='2.10', requires='>=36'),
    torchvision('0.26.0', pytorch='2.11', requires='>=36'),

    # JetPack 4
    torchvision('0.11.1', pytorch='1.10', requires='==32.*'),
    torchvision('0.10.0', pytorch='1.9', requires='==32.*'),
]
