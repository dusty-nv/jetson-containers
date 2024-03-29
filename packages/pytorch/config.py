from jetson_containers import CUDA_ARCHITECTURES


def pytorch_pip(version, requires, default=False, alias=None):
    """
    Install PyTorch from pip server with Dockerfile.pip
    """
    pkg = package.copy()
    
    pkg['name'] = f'pytorch:{version}'
    pkg['alias'] = [f'torch:{version}']
    
    if default:
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

    pkg['requires'] = requires
    
    return pkg
    
    
def pytorch_whl(version, whl, url, requires, default=False, alias=None):
    """
    Download & install PyTorch wheel with Dockerfile
    """
    pkg = package.copy()
    
    pkg['name'] = f'pytorch:{version}'
    pkg['alias'] = [f'torch:{version}']
    
    if default:
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


def pytorch_build(version, dockerfile='Dockerfile.builder', build_env_variables=None, depends=None, requires=None, suffix=None, default=False, alias=None):
    """
    Build PyTorch using the Dockerfile.builder
    """
    pkg = package.copy()
    
    pkg['name'] = f'pytorch:{version}'
    pkg['alias'] = f'torch:{version}'
    
    if suffix:
        pkg['name'] += '-' + suffix
        pkg['alias'] += '-' + suffix
    
    pkg['alias'] = [pkg['alias']]
    
    if default:
        pkg['alias'].extend(['pytorch', 'torch'])
        
    if alias:
        pkg['alias'].append(alias)

    pkg['dockerfile'] = dockerfile

    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {
        'PYTORCH_BUILD_VERSION': version,
        'PYTORCH_BUILD_NUMBER': '1',
        'PYTORCH_BUILD_EXTRA_ENV': build_env_variables,
        'TORCH_CUDA_ARCH_ARGS': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]), # retained as $TORCH_CUDA_ARCH_LIST
    }
    
    if depends:
        pkg['depends'] = pkg['depends'].copy().extend(depends)

    if requires:
        pkg['requires'] = requires
    
    return pkg


package = [
    # JetPack 6
    pytorch_pip('2.2', '==36.*', default=False),
    pytorch_pip('2.1', '==36.*', default=True),

    #pytorch_whl('2.1', 'torch-2.1.0-cp310-cp310-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/0h6tk4msrl9xz3evft9t0mpwwwkw7a32.whl', '==36.*', default=False),
    
    # JetPack 5
    pytorch_whl('2.1', 'torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl', '==35.*'),
    pytorch_whl('2.0', 'torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl', '==35.*', default=True),
    pytorch_whl('1.13', 'torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl', '==35.*'),
    pytorch_whl('1.12', 'torch-1.12.0a0+8a1a93a9.nv22.5-cp38-cp38-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+8a1a93a9.nv22.5-cp38-cp38-linux_aarch64.whl', '==35.*'),
    pytorch_whl('1.11', 'torch-1.11.0-cp38-cp38-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl', '==35.*'),
    
    # JetPack 4
    #pytorch('1.11', 'torch-1.11.0a0+17540c5-cp36-cp36m-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp36-cp36m-linux_aarch64.whl', '==32.*'),  # (built without LAPACK support)
    pytorch_whl('1.10', 'torch-1.10.0-cp36-cp36m-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl', '==32.*', default=True),
    pytorch_whl('1.9', 'torch-1.9.0-cp36-cp36m-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl', '==32.*'),

    # Build from source
    #pytorch_build('2.0', suffix='distributed', requires='==35.*'),            
    #pytorch_build('2.1', suffix='distributed', requires='==35.*', alias='pytorch:distributed'),        
    #pytorch_build('2.1', suffix='builder', requires='==36.*'),
]
