from packaging.version import Version
from ..pytorch.version import PYTORCH_VERSION

def torchvision(version, pytorch=None, requires=None):
    pkg = package.copy()
    
    pkg['name'] = f"torchvision:{version.split('-')[0]}"  # remove any -rc* suffix
    
    if pytorch:
        pkg['depends'] = [f"pytorch:{pytorch}" if x=='pytorch' else x for x in pkg['depends']]
    else:
        pytorch = PYTORCH_VERSION  
        
    if requires:
        pkg['requires'] = requires
    
    if not isinstance(pytorch, Version):
        pytorch = Version(pytorch)
        
    if pytorch == PYTORCH_VERSION:
        pkg['alias'] = 'torchvision'
     
    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {
        'TORCHVISION_VERSION': version,
    }

    return pkg
    
 
package = [
    # JetPack 6
    torchvision('0.16.2', pytorch='2.1', requires='==36.*'),
    torchvision('0.17.2', pytorch=PYTORCH_VERSION if PYTORCH_VERSION >= Version('2.2') else '2.2', requires='==36.*'),
    
    #torchvision('0.17.2', pytorch='2.2', requires='==36.*'),
    #torchvision('0.18.0-rc1', pytorch='2.3', requires='==36.*'),
    
    # JetPack 5
    torchvision('0.15.1', pytorch='2.0', requires='==35.*'),
    torchvision('0.16.2', pytorch='2.1', requires='==35.*'),
    
    # JetPack 4
    torchvision('0.11.1', pytorch='1.10', requires='==32.*'),
    torchvision('0.10.0', pytorch='1.9', requires='==32.*'),
]
