import os
from packaging.version import Version

from jetson_containers import (
    L4T_VERSION, JETPACK_VERSION, CUDA_VERSION,
    CUDA_ARCHITECTURES, LSB_RELEASE, IS_SBSA, IS_TEGRA,
    SYSTEM_ARM, DOCKER_ARCH, package_requires
)

def cuDSS(version, url, requires=None, default=False):
    """
    Container for cuDSS (Direct Sparse Solver)
    """
    pkg = package.copy()

    pkg['name'] = f'cudss:{version}'
    pkg['build_args'] = {'CUDSS_URL': url}

    if default:
        pkg['alias'] = ['cudss']

    if requires:
        pkg['requires'] = requires

    return pkg


if IS_TEGRA:
    package = [
        cuDSS('0.6',
            'https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb',
            requires=['tegra-aarch64', '22.04'],
            default=True,
        ),
        cuDSS('0.6',
            'https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2404-0.6.0_0.6.0-1_arm64.deb',
            requires=['aarch64', '24.04'],
            default=True,
        ),
    ]
elif IS_SBSA:
    package = [
        cuDSS('0.6',
              'https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-repo-ubuntu2404-0.6.0_0.6.0-1_arm64.deb',
              requires=['aarch64', '22.04'],
              default=True,
              ),
        cuDSS('0.6',
              'https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-repo-ubuntu2404-0.6.0_0.6.0-1_arm64.deb',
              requires=['aarch64', '24.04'],
              default=True,
              ),
    ]
else:
    package = [
        cuDSS('0.6',
              'https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-repo-ubuntu2204-0.6.0_0.6.0-1_amd64.deb',
              requires=['x86_64', '22.04'],
              default=True,
              ),
        cuDSS('0.6',
              'https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-repo-ubuntu2404-0.6.0_0.6.0-1_amd64.deb',
              requires=['x86_64', '24.04'],
              default=True,
              ),
    ]

