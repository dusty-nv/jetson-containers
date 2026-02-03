from jetson_containers import update_dependencies
from packaging.version import Version
from ..pytorch.version import PYTORCH_VERSION

def torchtext(version, pytorch=None, requires=None):
    pkg = package.copy()

    pkg['name'] = f"torchtext:{version.split('-')[0]}"  # remove any -rc* suffix

    if pytorch:
        pkg['depends'] = update_dependencies(pkg['depends'], f"pytorch:{pytorch}")
    else:
        pytorch = PYTORCH_VERSION

    if requires:
        pkg['requires'] = requires

    if len(version.split('.')) < 3:
        version = version + '.0'

    pkg['build_args'] = {
        'TORCHTEXT_VERSION': version,
    }

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if not isinstance(pytorch, Version):
        pytorch = Version(pytorch)

    if pytorch == PYTORCH_VERSION:
        pkg['alias'] = 'torchtext'
        builder['alias'] = 'torchtext:builder'

    return pkg, builder


package = [
    # JetPack 5/6 and x86
    torchtext('0.20.1', pytorch='2.9.1', requires='>=34'),
    torchtext('0.20.2', pytorch='2.10', requires='>=34'),
    torchtext('0.21.0', pytorch='2.11', requires='>=34'),

]
