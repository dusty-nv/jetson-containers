from jetson_containers import update_dependencies
from packaging.version import Version
from ..pytorch.version import PYTORCH_VERSION

def torchaudio(version, pytorch=None, requires=None):
    pkg = package.copy()

    pkg['name'] = f"torchaudio:{version.split('-')[0]}"  # remove any -rc* suffix

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
        'TORCHAUDIO_VERSION': version,
    }

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if pytorch == PYTORCH_VERSION:
        pkg['alias'] = 'torchaudio'
        builder['alias'] = 'torchaudio:builder'

    return pkg, builder

package = [
    # JetPack 5/6
    torchaudio('2.2.2', pytorch='2.2', requires='>=35'),
    torchaudio('2.7.0', pytorch='2.7', requires='==36.*'),
    torchaudio('2.9.1', pytorch='2.9.1', requires='>=36'),
    torchaudio('2.10.0', pytorch='2.10', requires='>=36'),
    torchaudio('2.11.0', pytorch='2.11', requires='>=36'),

    # JetPack 4
    torchaudio('0.10.0', pytorch='1.10', requires='==32.*'),
    torchaudio('0.9.0', pytorch='1.9', requires='==32.*'),
]
