from jetson_containers import L4T_VERSION, CUDA_VERSION, SYSTEM_ARM, update_dependencies, package_requires, IS_TEGRA, IS_SBSA
from packaging.version import Version

import os

# Define the default CUDNN_VERSION either from environment variable or
# as to what version of cuDNN was released with that version of CUDA
if 'CUDNN_VERSION' in os.environ and len(os.environ['CUDNN_VERSION']) > 0:
    CUDNN_VERSION = Version(os.environ['CUDNN_VERSION'])
elif SYSTEM_ARM:
    if L4T_VERSION.major >= 38:
        if CUDA_VERSION >= Version('13.0'):
            CUDNN_VERSION = Version('9.12')
    elif L4T_VERSION.major >= 36:
        if CUDA_VERSION >= Version('12.9'):
            CUDNN_VERSION = Version('9.13')
        elif CUDA_VERSION >= Version('12.8'):
            CUDNN_VERSION = Version('9.8')
        elif CUDA_VERSION == Version('12.6'):
            CUDNN_VERSION = Version('9.3')
        elif CUDA_VERSION == Version('12.4'):
            CUDNN_VERSION = Version('9.0')
        else:
            CUDNN_VERSION = Version('8.9')
    elif L4T_VERSION.major >= 34:
        CUDNN_VERSION = Version('8.6')
    elif L4T_VERSION.major >= 32:
        CUDNN_VERSION = Version('8.2')
else:
    CUDNN_VERSION = Version('9.12') # x86_64

def cudnn_package(version, url, deb=None, packages=None, cuda=None, requires=None):
    """
    Generate containers for a particular version of cuDNN installed from debian packages
    """
    if not deb:
        deb = url.split('/')[-1].split('_')[0]

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

    package_requires(cudnn, system_arch='aarch64') # default to aarch64

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

CUDNN_URL='https://developer.download.nvidia.com/compute/cudnn'
IS_CONFIG='package' in globals()  # CUDNN_VERSION gets imported by other packages

if IS_TEGRA and IS_CONFIG:
    package = [
        # JetPack 7
        cudnn_package('9.12.0',f'{CUDNN_URL}/9.12.0/local_installers/cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_arm64.deb', cuda='13.0', requires='>=38', packages="libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-samples"),

        # JetPack 6
        cudnn_package('8.9','https://nvidia.box.com/shared/static/ht4li6b0j365ta7b76a6gw29rk5xh8cy.deb', 'cudnn-local-tegra-repo-ubuntu2204-8.9.4.25', cuda='12.2', requires='==36.*'),
        cudnn_package('9.0',f'{CUDNN_URL}/9.0.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.0.0_1.0-1_arm64.deb', cuda='12.4', requires='==36.*'),
        cudnn_package('9.3',f'{CUDNN_URL}/9.3.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb', cuda='12.6', requires='==36.*'),
        cudnn_package('9.4',f'{CUDNN_URL}/9.4.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.4.0_1.0-1_arm64.deb', cuda='12.6', requires='==36.*'),
        cudnn_package('9.8',f'{CUDNN_URL}/9.8.0/local_installers/cudnn-local-tegra-repo-ubuntu2404-9.8.0_1.0-1_arm64.deb', cuda='12.8', requires='>=36', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
        cudnn_package('9.10',f'{CUDNN_URL}/9.10.2/local_installers/cudnn-local-tegra-repo-ubuntu2404-9.10.2_1.0-1_arm64.deb', cuda='12.9', requires='>=36', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
        cudnn_package('9.11.0',f'{CUDNN_URL}/9.11.0/local_installers/cudnn-local-tegra-repo-ubuntu2404-9.11.0_1.0-1_arm64.deb', cuda='12.9', requires='>=36', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
        cudnn_package('9.12.0',f'{CUDNN_URL}/9.12.0/local_installers/cudnn-local-tegra-repo-ubuntu2404-9.12.0_1.0-1_arm64.deb', cuda='12.9', requires='>=36', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
        cudnn_package('9.13.0',f'{CUDNN_URL}/9.13.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.13.0_1.0-1_arm64.deb', cuda='12.9', requires='>=36', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
        # JetPack 4-5 (cuDNN installed in base container)
        cudnn_builtin(requires='<36', default=True),
    ]

elif IS_SBSA and IS_CONFIG:
    # sbsa
    package = [
        cudnn_package('9.8',f'{CUDNN_URL}/9.8.0/local_installers/cudnn-local-repo-ubuntu2404-9.8.0_1.0-1_arm64.deb', cuda='12.8', requires='aarch64', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
        cudnn_package('9.10',f'{CUDNN_URL}/9.10.2/local_installers/cudnn-local-repo-ubuntu2404-9.10.2_1.0-1_arm64.deb', cuda='12.9', requires='aarch64', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
        cudnn_package('9.11.0',f'{CUDNN_URL}/9.11.0/local_installers/cudnn-local-repo-ubuntu2404-9.11.0_1.0-1_arm64.deb', cuda='13.0', requires='aarch64', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
        cudnn_package('9.12.0',f'{CUDNN_URL}/9.12.0/local_installers/cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_arm64.deb', cuda='13.0', requires='aarch64', packages="libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-samples"),

    ]
elif IS_CONFIG:
    # x86_64
    package = [
        cudnn_package('9.8',f'{CUDNN_URL}/9.8.0/local_installers/cudnn-local-repo-ubuntu2404-9.8.0_1.0-1_amd64.deb', cuda='12.8', requires='x86_64', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
        cudnn_package('9.10',f'{CUDNN_URL}/9.10.2/local_installers/cudnn-local-repo-ubuntu2404-9.10.2_1.0-1_amd64.deb', cuda='12.9', requires='x86_64', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
        cudnn_package('9.11.0',f'{CUDNN_URL}/9.11.0/local_installers/cudnn-local-repo-ubuntu2404-9.11.0_1.0-1_amd64.deb', cuda='13.0', requires='x86_64', packages="libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"),
        cudnn_package('9.12.0',f'{CUDNN_URL}/9.12.0/local_installers/cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb', cuda='13.0', requires='x86_64', packages="libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-samples"),
    ]
    
