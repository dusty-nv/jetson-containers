import os

from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES, CUDA_VERSION


def cuda_package(version, url, deb, packages=None, requires=None, default=False):
    """
    Generate containers for a particular version of CUDA installed from debian packages
    This will download & install the specified packages (by default the full CUDA Toolkit) 
    from a .deb URL from developer.nvidia.com/cuda-downloads (the `aarch64-jetson` versions)
    """
    if not packages:
        packages = os.environ.get('CUDA_PACKAGES', 'cuda-toolkit*')
    
    cuda = package.copy()
    
    cuda['name'] = f'cuda:{version}'
    
    cuda['build_args'] = {
        'CUDA_URL': url,
        'CUDA_DEB': deb,
        'CUDA_PACKAGES': packages,
    }

    if default:
        cuda['alias'] = 'cuda'
    
    if requires:
        cuda['requires'] = requires
        
    if 'toolkit' in packages or 'dev' in packages:
        cuda['depends'] = ['build-essential']
        
    return cuda


def cuda_builtin(version, requires=None, default=False):
    """
    Backwards-compatability for when CUDA already installed in base container (like l4t-jetpack)
    This will just retag the base, marking CUDA dependency as satisfied in any downstream containers.
    """
    passthrough = package.copy()

    if not isinstance(version, str):
        version = f'{version.major}.{version.minor}'
 
    passthrough['name'] = f'cuda:{version}'

    del passthrough['dockerfile']
    
    if default:
        passthrough['alias'] = 'cuda'
        
    if requires:
        passthrough['requires'] = requires
        
    passthrough['depends'] = ['build-essential']
    
    return passthrough


def cuda_samples(version, requires, default=False):
    """
    Generates container that installs/builds the CUDA samples
    """
    samples = package.copy()
    
    if not isinstance(version, str):
        version = f'{version.major}.{version.minor}'
        
    samples['name'] = f'cuda:{version}-samples'
    samples['dockerfile'] = 'Dockerfile.samples'
    samples['notes'] = "CUDA samples from https://github.com/NVIDIA/cuda-samples installed under /opt/cuda-samples"
    samples['test'] = 'test_samples.sh'
    samples['depends'] = [f'cuda:{version}', 'cmake']
    
    samples['build_args'] = {'CUDA_BRANCH': 'v' + version}
    
    if default:
        samples['alias'] = 'cuda:samples'
        
    if requires:
        samples['requires'] = requires
        
    return samples
    
    
package = [
    
    # JetPack 6
    cuda_package('12.2', 'https://nvidia.box.com/shared/static/uvqtun1sc0bq76egarc8wwuh6c23e76e.deb', 'cuda-tegra-repo-ubuntu2204-12-2-local', requires='==36.*', default=True), 
    cuda_samples('12.2', requires='==36.*', default=True),
    
    # JetPack 5
    cuda_package('12.2', 'https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-tegra-repo-ubuntu2004-12-2-local_12.2.2-1_arm64.deb', 'cuda-tegra-repo-ubuntu2004-12-2-local', requires='==35.*', default=False),
    cuda_package('11.8', 'https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb', 'cuda-tegra-repo-ubuntu2004-11-8-local', requires='==35.*', default=False),
    
    cuda_samples('12.2', requires='==35.*', default=False),
    cuda_samples('11.8', requires='==35.*', default=False),
    
    # JetPack 4-5 (CUDA installed in base container)
    cuda_builtin(CUDA_VERSION, requires='<36', default=True),
    cuda_samples(CUDA_VERSION, requires='<36', default=True),
]

