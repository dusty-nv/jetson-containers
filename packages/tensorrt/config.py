import os

from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES, CUDA_VERSION


def tensorrt_package(version, url, deb, packages=None, requires=None, default=False):
    """
    Generate containers for a particular version of TensorRT installed from debian packages
    """
    if not packages:
        packages = os.environ.get('TENSORRT_PACKAGES', 'tensorrt tensorrt-libs python3-libnvinfer-dev')
    
    tensorrt = package.copy()
    
    tensorrt['name'] = f'tensorrt:{version}'

    tensorrt['build_args'] = {
        'TENSORRT_URL': url,
        'TENSORRT_DEB': deb,
        'TENSORRT_PACKAGES': packages,
    }

    if default:
        tensorrt['alias'] = 'tensorrt'
    
    if requires:
        tensorrt['requires'] = requires

    return tensorrt


def tensorrt_builtin(version=None, requires=None, default=False):
    """
    Backwards-compatability for when TensorRT already installed in base container (like l4t-jetpack)
    """
    passthrough = package.copy()

    if version is not None:
        if not isinstance(version, str):
            version = f'{version.major}.{version.minor}'
           
        if default:
            passthrough['alias'] = 'tensorrt'  
            
        passthrough['name'] += f':{version}'
     
    if requires:
        passthrough['requires'] = requires
        
    del passthrough['dockerfile']
    return passthrough

    
package = [
    
    # JetPack 6
    tensorrt_package('8.6', 'https://nvidia.box.com/shared/static/hmwr57hm88bxqrycvlyma34c3k4c53t9.deb', 'nv-tensorrt-local-repo-l4t-8.6.2-cuda-12.2', requires='==36.*', default=True), 

    # JetPack 4-5 (TensorRT installed in base container)
    tensorrt_builtin(requires='<36', default=True),
]

