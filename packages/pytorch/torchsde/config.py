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
    # JetPack 5/6 and x86
    torchsde('0.2.0', pytorch='2.0', requires='==35.*'),
    torchsde('0.2.1', pytorch='2.1', requires='>=35'),
    torchsde('0.2.2', pytorch='2.2', requires='>=35'),
    torchsde('0.2.3', pytorch='2.3', requires='==36.*'),
    torchsde('0.2.4', pytorch='2.4', requires='==36.*'),
    torchsde('0.2.5', pytorch='2.5', requires='==36.*'),
    torchsde('0.2.6', pytorch='2.6', requires='==36.*'),
    torchsde('0.2.7', pytorch='2.7', requires='==36.*'),
    torchsde('0.2.8', pytorch='2.8', requires='==36.*'),

    # JetPack 4
    torchsde('0.2.7', pytorch='1.10', requires='==32.*'),
    torchsde('0.2.7', pytorch='1.9', requires='==32.*'),
]
