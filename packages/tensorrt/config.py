from jetson_containers import L4T_VERSION, CUDA_VERSION, update_dependencies
from packaging.version import Version

import os

package['depends'] = ['cuda', 'cudnn', 'python']
package['test'] = ['test.sh']

if 'TENSORRT_VERSION' in os.environ and len(os.environ['TENSORRT_VERSION']) > 0:
    TENSORRT_VERSION = Version(os.environ['TENSORRT_VERSION'])
else:
    if L4T_VERSION.major >= 36:
        if CUDA_VERSION >= Version('12.4'):
            TENSORRT_VERSION = Version('10.0')
        else:
            TENSORRT_VERSION = Version('8.6')
    elif L4T_VERSION.major >= 34:
        TENSORRT_VERSION = Version('8.5')
    elif L4T_VERSION.major >= 32:
        TENSORRT_VERSION = Version('8.2')
        
        
def tensorrt_deb(version, url, deb, cudnn=None, packages=None, requires=None):
    """
    Generate containers for a particular version of TensorRT installed from debian packages
    """
    if not packages:
        packages = os.environ.get('TENSORRT_PACKAGES', 'tensorrt tensorrt-libs python3-libnvinfer-dev')
    
    tensorrt = package.copy()
    
    tensorrt['name'] = f'tensorrt:{version}'
    tensorrt['dockerfile'] = 'Dockerfile.deb'
    
    tensorrt['build_args'] = {
        'TENSORRT_URL': url,
        'TENSORRT_DEB': deb,
        'TENSORRT_PACKAGES': packages,
    }

    if Version(version) == TENSORRT_VERSION:
        tensorrt['alias'] = 'tensorrt'
    
    if cudnn:
        tensorrt['depends'] = update_dependencies(tensorrt['depends'], f"cudnn:{cudnn}")
         
    if requires:
        tensorrt['requires'] = requires

    return tensorrt


def tensorrt_tar(version, url, cudnn=None, requires=None):
    """
    Generate containers for a particular version of TensorRT installed from tar.gz file
    """
    tensorrt = package.copy()
    
    tensorrt['name'] = f'tensorrt:{version}'

    tensorrt['dockerfile'] = 'Dockerfile.tar'
    tensorrt['build_args'] = {'TENSORRT_URL': url}

    if Version(version) == TENSORRT_VERSION:
        tensorrt['alias'] = 'tensorrt'
    
    if cudnn:
        tensorrt['depends'] = update_dependencies(tensorrt['depends'], f"cudnn:{cudnn}")
         
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
        
    #del passthrough['dockerfile']
    return passthrough

    
package = [
    # JetPack 6
    tensorrt_deb('8.6', 'https://nvidia.box.com/shared/static/hmwr57hm88bxqrycvlyma34c3k4c53t9.deb', 'nv-tensorrt-local-repo-l4t-8.6.2-cuda-12.2', cudnn='8.9', requires=['==r36.*', '==cu122']), 
    #tensorrt_tar('9.3', 'https://nvidia.box.com/shared/static/fp3o14iq7qbm67qjuqivdrdch7009axu.gz', cudnn='8.9', requires=['==r36.*', '==cu122']), 
    tensorrt_tar('10.0', 'https://nvidia.box.com/shared/static/n00by5242vvf800jbhzjioj77vd7fqu7.gz', cudnn='9.0', requires=['==r36.*', '==cu124']), 
    
    # JetPack 4-5 (TensorRT installed in base container)
    tensorrt_builtin(requires='<36', default=True),
]

