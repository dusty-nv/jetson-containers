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
        
    if pytorch == PYTORCH_VERSION:
        pkg['alias'] = 'torchaudio'

    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {
        'TORCHAUDIO_VERSION': version,
    }

    return pkg

package = [
    # JetPack 6
    torchaudio('2.1.0', pytorch='2.1', requires='==36.*'),
    torchaudio('2.2.2', pytorch='2.2', requires='==36.*'),
    torchaudio('2.3.0-rc2', pytorch='2.3', requires='==36.*'),
    
    # JetPack 5
    torchaudio('2.0.1', pytorch='2.0', requires='==35.*'),
    torchaudio('2.1.0', pytorch='2.1', requires='==35.*'),
    
    # JetPack 4
    torchaudio('0.10.0', pytorch='1.10', requires='==32.*'),
    torchaudio('0.9.0', pytorch='1.9', requires='==32.*'),
]
