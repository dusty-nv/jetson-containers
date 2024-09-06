from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES
from packaging.version import Version

from .version import JAX_VERSION


def jax_pip(version, requires=None, alias=None):
    """
    Install JAX from pip server with Dockerfile.pip
    """
    pkg = package.copy()
    
    short_version = Version(version.split('-')[0])  # remove any -rc* suffix
    short_version = f"{short_version.major}.{short_version.minor}"
        
    pkg['name'] = f'jax:{short_version}'    
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
        pkg['build_args']['USE_NCCL'] = 1
        
    if requires:
        pkg['requires'] = requires
    
    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    pkg['alias'] = [f'jax:{short_version}']
    builder['alias'] = [f'jax:{short_version}-builder']
    
    if Version(short_version) == JAX_VERSION:
        pkg['alias'].extend(['jax'])
        builder['alias'].extend(['jax:builder'])
        
    if alias:
        pkg['alias'].append(alias)

    return pkg, builder


def jax_whl(version, whl, url, requires, alias=None):
    """
    Download & install JAX wheel with Dockerfile
    """
    pkg = package.copy()
    
    pkg['name'] = f'jax:{version}'
    pkg['alias'] = [f'jax:{version}']
    
    if Version(version) == JAX_VERSION:
        pkg['alias'].extend(['jax'])
    
    if alias:
        pkg['alias'].append(alias)
        
    pkg['build_args'] = {
        'JAX_WHL': whl,
        'JAX_URL': url,
        'JAX_CUDA_ARCH_ARGS': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])  # retained as $JAX_CUDA_ARCH_LIST
    }

    pkg['requires'] = requires
    
    return pkg


package = [
    # JetPack 5/6
    jax_pip('0.4', requires='==35.*'),
    jax_pip('0.4.1', requires='>=35'),
    jax_pip('0.4.2', requires='>=35'),
    jax_pip('0.4.3', requires='==36.*'),
    jax_pip('0.4.4', requires='==36.*'),

    # JetPack 4
    jax_whl('0.3.10', 'jax-0.3.10-cp36-cp36m-linux_aarch64.whl', 'https://path_to_jax_whl/jax-0.3.10-cp36-cp36m-linux_aarch64.whl', '==32.*'),
    jax_whl('0.3.9', 'jax-0.3.9-cp36-cp36m-linux_aarch64.whl', 'https://path_to_jax_whl/jax-0.3.9-cp36-cp36m-linux_aarch64.whl', '==32.*'),
]
