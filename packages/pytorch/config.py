from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES
from packaging.version import Version

from .version import PYTORCH_VERSION
    

def pytorch_pip(version, requires=None, alias=None):
    """
    Install PyTorch from pip server with Dockerfile.pip
    """
    pkg = package.copy()
    
    short_version = Version(version.split('-')[0]) # remove any -rc* suffix
    short_version = f"{short_version.major}.{short_version.minor}"
        
    pkg['name'] = f'pytorch:{short_version}'
    pkg['alias'] = [f'torch:{short_version}']
    
    if Version(short_version) == PYTORCH_VERSION:
        pkg['alias'].extend(['pytorch', 'torch'])
    
    if alias:
        pkg['alias'].append(alias)
    
    pkg['dockerfile'] = 'Dockerfile.pip'
    
    if len(version.split('.')) < 3:
        build_version = version + '.0'
    else:
        build_version = version
        
    pkg['build_args'] = {
        'TORCH_CUDA_ARCH_ARGS': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]), # retained as $TORCH_CUDA_ARCH_LIST
        'TORCH_VERSION': version,
        'PYTORCH_BUILD_VERSION': build_version,
    }

    if requires:
        pkg['requires'] = requires
    
    return pkg
    
    
def pytorch_whl(version, whl, url, requires, alias=None):
    """
    Download & install PyTorch wheel with Dockerfile
    """
    pkg = package.copy()
    
    pkg['name'] = f'pytorch:{version}'
    pkg['alias'] = [f'torch:{version}']
    
    if Version(version) == PYTORCH_VERSION:
        pkg['alias'].extend(['pytorch', 'torch'])
    
    if alias:
        pkg['alias'].append(alias)
        
    pkg['build_args'] = {
        'PYTORCH_WHL': whl,
        'PYTORCH_URL': url,
        'TORCH_CUDA_ARCH_ARGS': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]) # retained as $TORCH_CUDA_ARCH_LIST
    }

    pkg['requires'] = requires
    
    return pkg


package = [
    # JetPack 6
    pytorch_pip('2.1', requires='==36.*'),
    pytorch_pip('2.2', requires='==36.*'),
    pytorch_pip('2.3.0-rc6', requires='==36.*'),

    # JetPack 5
    pytorch_pip('2.1', requires='==35.*'),
    pytorch_pip('2.0', requires='==35.*'),

    # JetPack 4
    pytorch_whl('1.10', 'torch-1.10.0-cp36-cp36m-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl', '==32.*'),
    pytorch_whl('1.9', 'torch-1.9.0-cp36-cp36m-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl', '==32.*'),
]
