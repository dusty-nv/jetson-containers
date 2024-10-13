from jetson_containers import update_dependencies
from packaging.version import Version
from ..pytorch.version import PYTORCH_VERSION

def torchao(version, pytorch=None, requires=None):
    pkg = package.copy()
    
    pkg['name'] = f"torchao:{version.split('-')[0]}"  # remove any -rc* suffix
    
    if pytorch:
        pkg['depends'] = update_dependencies(pkg['depends'], f"pytorch:{pytorch}")
    else:
        pytorch = PYTORCH_VERSION  
        
    if requires:
        pkg['requires'] = requires
   
    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {
        'TORCHAO_VERSION': version,
    }
    
    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    if not isinstance(pytorch, Version):
        pytorch = Version(pytorch)

    if pytorch == PYTORCH_VERSION:
        pkg['alias'] = 'torchao'
        builder['alias'] = 'torchao:builder'

    return pkg, builder
    
 
package = [
    # JetPack 5/6
    torchao('0.6.0', pytorch='2.4', requires='==36.*'),
    torchao('0.7.0', pytorch='2.5', requires='==36.*'),
]
