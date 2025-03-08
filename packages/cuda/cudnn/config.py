
from jetson_containers import L4T_VERSION, CUDA_VERSION, update_dependencies
from packaging.version import Version

import os

if 'CUDNN_VERSION' in os.environ and len(os.environ['CUDNN_VERSION']) > 0:
    CUDNN_VERSION = Version(os.environ['CUDNN_VERSION'])
else:
    if L4T_VERSION.major >= 36:
        if CUDA_VERSION >= Version('12.8'):
            CUDNN_VERSION = Version('9.8')
        elif CUDA_VERSION == Version('12.6'):
            CUDNN_VERSION = Version('9.4')
        elif CUDA_VERSION == Version('12.4'):
            CUDNN_VERSION = Version('9.0')
        else:
            CUDNN_VERSION = Version('8.9')
    elif L4T_VERSION.major >= 34:
        CUDNN_VERSION = Version('8.6')
    elif L4T_VERSION.major >= 32:
        CUDNN_VERSION = Version('8.2')

#print(f"-- CUDNN_VERSION={CUDNN_VERSION}")
       
def cudnn_package(version, url, deb, packages=None, cuda=None, requires=None):
    """
    Generate containers for a particular version of cuDNN installed from debian packages
    """
    if not packages:
        packages = os.environ.get('CUDNN_PACKAGES', 'libcudnn*-dev libcudnn*-samples')
    
    cudnn = package.copy()
    
    cudnn['name'] = f'cudnn:{version}'
    
    cudnn['build_args'] = {
        'CUDNN_URL': url,
        'CUDNN_DEB': deb,
        'CUDNN_PACKAGES': packages,
    }

    if Version(version) == CUDNN_VERSION:
        cudnn['alias'] = 'cudnn'
    
    if cuda:
        cudnn['depends'] = update_dependencies(cudnn['depends'], f"cuda:{cuda}")
        
    if requires:
        cudnn['requires'] = requires

    return cudnn

def cudnn_builtin(version=None, requires=None, default=False):
    """
    Backwards-compatability for when cuDNN already installed in base container (like l4t-jetpack)
    """
    passthrough = package.copy()

    if version is not None:
        if not isinstance(version, str):
            version = f'{version.major}.{version.minor}'
           
        if default:
            passthrough['alias'] = 'cudnn'  
            
        passthrough['name'] += f':{version}'
     
    if requires:
        passthrough['requires'] = requires
        
    del passthrough['dockerfile']
    passthrough['depends'] = ['cuda']
    
    return passthrough

    
package = [
    
    # JetPack 6
    cudnn_package('8.9', 'https://nvidia.box.com/shared/static/ht4li6b0j365ta7b76a6gw29rk5xh8cy.deb', 'cudnn-local-tegra-repo-ubuntu2204-8.9.4.25', cuda='12.2', requires='==36.*'), 
    cudnn_package('9.0', 'https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.0.0_1.0-1_arm64.deb', 'cudnn-local-tegra-repo-ubuntu2204-9.0.0', cuda='12.4', requires='==36.*'),
    cudnn_package('9.4', 'https://developer.download.nvidia.com/compute/cudnn/9.4.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.4.0_1.0-1_arm64.deb', 'cudnn-local-tegra-repo-ubuntu2204-9.4.0', cuda='12.6', requires='==36.*'),
    cudnn_package('9.8', 'https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-tegra-repo-ubuntu2404-9.8.0_1.0-1_arm64.deb', 'cudnn-local-tegra-repo-ubuntu2404-9.8.0', cuda='12.8', requires='==36.*', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
    # JetPack 4-5 (cuDNN installed in base container)
    cudnn_builtin(requires='<36', default=True),
]

