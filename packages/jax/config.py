from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

def jax_pip(version, requires=None, alias=None, default=False):
    """
    Install JAX from pip server with Dockerfile.pip
    """
    pkg = package.copy()
        
    pkg['name'] = f'jax:{version}'    
    pkg['dockerfile'] = 'Dockerfile.pip'
    
    if len(version.split('.')) < 3:
        build_version = version + '.0'
    else:
        build_version = version
        
    pkg['build_args'] = {
        'JAX_CUDA_ARCH_ARGS': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]),  # retained as $JAX_CUDA_ARCH_LIST
        'JAX_VERSION': version,
        'JAX_BUILD_VERSION': build_version,
    }
    
    if L4T_VERSION.major >= 36:
        pkg['build_args']['ENABLE_NCCL'] = 1
   
    if requires:
        pkg['requires'] = requires
    
    
    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    if default:
        pkg['alias'] = 'jax'
        builder['alias'] = 'jax:builder'

    return pkg, builder


def jax_whl(version, whl, url, requires, alias=None):
    """
    Download & install JAX wheel with Dockerfile
    """
    pkg = package.copy()
    
    pkg['name'] = f'jax:{version}'
        
    pkg['build_args'] = {
        'JAX_WHL': whl,
        'JAX_URL': url,
        'JAX_CUDA_ARCH_ARGS': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])  # retained as $JAX_CUDA_ARCH_LIST
    }

    pkg['requires'] = requires
    
    return pkg


package = [
    # JetPack 5/6
    jax_pip('0.4.31', requires='==35.*'),
    jax_pip('0.4.31', requires='>=35'),
    jax_pip('0.4.31', requires='>=35'),
    jax_pip('0.4.31', requires='==36.*'),
    jax_pip('0.4.31', requires='==36.*', default=True),
]
