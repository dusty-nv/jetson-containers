
from jetson_containers import L4T_VERSION, CUDA_VERSION, update_dependencies
from packaging.version import Version

import os

if 'OPTIX_VERSION' in os.environ and len(os.environ['OPTIX_VERSION']) > 0:
    OPTIX_VERSION = Version(os.environ['OPTIX_VERSION'])
else:
    if L4T_VERSION.major >= 36:
        if CUDA_VERSION >= Version('12.8'):
            OPTIX_VERSION = Version('9.0.0')
        else:
            OPTIX_VERSION = Version('9.0.0')
    elif L4T_VERSION.major >= 34:
        OPTIX_VERSION = Version('9.0.0')
    elif L4T_VERSION.major >= 32:
        OPTIX_VERSION = Version('9.0.0')

#print(f"-- OPTIX_VERSION={OPTIX_VERSION}")
       
def optix_package(version, url, deb, packages=None, cuda=None, requires=None):
    """
    Generate containers for a particular version of optix installed from debian packages
    """
       
    optix = package.copy()
    
    optix['name'] = f'optix:{version}'
    
    optix['build_args'] = {
        'OPTIX_URL': url,
    }

    if Version(version) == OPTIX_VERSION:
        optix['alias'] = 'optix'
    
    if cuda:
        optix['depends'] = update_dependencies(optix['depends'], f"cuda:{cuda}")
        
    if requires:
        optix['requires'] = requires

    return optix
   
package = [
    
    # JetPack 6
    optix_package('9.0.0', 'https://developer.nvidia.com/downloads/designworks/optix/secure/9.0.0/nvidia-optix-sdk-9.0.0-linux64-aarch64.sh', cuda='12.6', requires='==36.*'),
    optix_package('9.0.0', 'https://developer.nvidia.com/downloads/designworks/optix/secure/9.0.0/nvidia-optix-sdk-9.0.0-linux64-aarch64.sh', cuda='12.8', requires='==36.*'),
]
