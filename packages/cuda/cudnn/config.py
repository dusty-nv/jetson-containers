import os

from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES, CUDA_VERSION


def cudnn_package(version, url, deb, packages=None, depends=None, requires=None, default=False):
    """
    Generate containers for a particular version of cuDNN installed from debian packages
    """
    if not packages:
        packages = os.environ.get('CUDNN_PACKAGES', 'libcudnn*-dev libcudnn*-samples')
    
    if not depends:
        depends = ['cuda']
        
    cudnn = package.copy()
    
    cudnn['name'] = f'cudnn:{version}'
    cudnn['depends'] = depends
    
    cudnn['build_args'] = {
        'CUDNN_URL': url,
        'CUDNN_DEB': deb,
        'CUDNN_PACKAGES': packages,
    }

    if default:
        cudnn['alias'] = 'cudnn'
    
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
    cudnn_package('8.9', 'https://nvidia.box.com/shared/static/ht4li6b0j365ta7b76a6gw29rk5xh8cy.deb', 'cudnn-local-tegra-repo-ubuntu2204-8.9.4.25', requires='==36.*', default=True), 

    # JetPack 4-5 (cuDNN installed in base container)
    cudnn_builtin(requires='<36', default=True),
]

