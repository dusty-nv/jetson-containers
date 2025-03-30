from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES, LSB_RELEASE
from packaging.version import Version

from .version import PYTORCH_VERSION
    

def pytorch_pip(version, requires=None, alias=None):
    """
    Install PyTorch from pip server with Dockerfile.pip
    """
    pkg = package.copy()
    
    short_version = Version(version.split('-')[0]) # remove any -rc* suffix
    short_version = f"{short_version.major}.{short_version.minor}"
    micro_version = Version(version).micro
    
    if micro_version > 0:
        short_version += f".{micro_version}"
            
    pkg['name'] = f'pytorch:{short_version}'    
    pkg['dockerfile'] = 'Dockerfile'
    
    if len(version.split('.')) < 3:
        build_version = version + '.0'
    else:
        build_version = version

    pkg['build_args'] = {
        'TORCH_CUDA_ARCH_ARGS': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]), # retained as $TORCH_CUDA_ARCH_LIST
        'TORCH_VERSION': version,
        'PYTORCH_BUILD_VERSION': build_version,
    }

    if Version(LSB_RELEASE) >= Version('24.04'):
        pkg['build_args']['USE_XNNPACK'] = 0

    if L4T_VERSION.major >= 36:
        pkg['build_args']['USE_NCCL'] = 1  # NCCL building only on JP6 and newer
        
    if Version(version) >= Version('2.3.1'): # begin disabling MPI with JP 6.1 since GLOO/NCCL is working
        pkg['build_args']['USE_MPI'] = 0     # and to eliminate security vulnerability from MPI packages
        
    if requires:
        pkg['requires'] = requires
    
    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    pkg['alias'] = [f'torch:{short_version}']
    builder['alias'] = [f'torch:{short_version}-builder']
    
    if Version(short_version) == PYTORCH_VERSION:
        pkg['alias'].extend(['pytorch', 'torch'])
        builder['alias'].extend(['pytorch:builder', 'torch:builder'])
        
    if alias:
        pkg['alias'].append(alias)

    return pkg, builder
    
    
def pytorch_wget(version, whl, url, requires, alias=None):
    """
    Download & install PyTorch wheel with Dockerfile
    """
    pkg = package.copy()
    
    pkg['name'] = f'pytorch:{version}'
    pkg['alias'] = [f'torch:{version}']
    pkg['Dockerfile'] = 'Dockerfile.wget'
    
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
    # JetPack 5/6
    pytorch_pip('2.0', requires='==35.*'),
    pytorch_pip('2.1', requires='>=35'),
    pytorch_pip('2.2', requires='>=35'),
    pytorch_pip('2.3.0', requires='==36.*'),
    pytorch_pip('2.3.1', requires='==36.*'),  # without OpenMPI
    pytorch_pip('2.4', requires='==36.*'),
    pytorch_pip('2.5', requires='==36.*'),    # without OpenMPI
    pytorch_pip('2.6', requires='==36.*'),    # without OpenMPI
    pytorch_pip('2.7', requires='==36.*'),    # without OpenMPI

    # JetPack 4
    pytorch_wget('1.10', 'torch-1.10.0-cp36-cp36m-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl', '==32.*'),
    pytorch_wget('1.9', 'torch-1.9.0-cp36-cp36m-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl', '==32.*'),
]
