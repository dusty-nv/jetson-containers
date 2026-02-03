from jetson_containers import update_dependencies
from packaging.version import Version
from ..pytorch.version import PYTORCH_VERSION

def torchsde(version, pytorch=None, requires=None):
    pkg = package.copy()

    pkg['name'] = f"torchsde:{version.split('-')[0]}"  # remove any -rc* suffix

    if pytorch:
        pkg['depends'] = update_dependencies(pkg['depends'], f"pytorch:{pytorch}")
    else:
        pytorch = PYTORCH_VERSION

    if requires:
        pkg['requires'] = requires

    if len(version.split('.')) < 3:
        version = version + '.0'

    pkg['build_args'] = {
        'TORCHSDE_VERSION': version,
    }

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if not isinstance(pytorch, Version):
        pytorch = Version(pytorch)

    if pytorch == PYTORCH_VERSION:
        pkg['alias'] = 'torchsde'
        builder['alias'] = 'torchsde:builder'

    return pkg, builder


package = [
    torchsde('2.2.2', pytorch='2.2', requires='>=35'),
    torchsde('2.7.0', pytorch='2.7', requires='==36.*'),
    torchsde('2.9.1', pytorch='2.9.1', requires='>=36'),
    torchsde('2.10.0', pytorch='2.10', requires='>=36'),
    torchsde('2.11.0', pytorch='2.11', requires='>=36'),

    # JetPack 4
    torchsde('0.10.0', pytorch='1.10', requires='==32.*'),
    torchsde('0.9.0', pytorch='1.9', requires='==32.*'),
]
