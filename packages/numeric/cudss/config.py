
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


package = [
    cuDSS('0.5',
        'https://developer.download.nvidia.com/compute/cudss/0.5.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.5.0_0.5.0-1_arm64.deb',
        requires=['aarch64', '22.04'],
        default=True,
    ),
    cuDSS('0.5',
        'https://developer.download.nvidia.com/compute/cudss/0.5.0/local_installers/cudss-local-tegra-repo-ubuntu2404-0.5.0_0.5.0-1_arm64.deb',
        requires=['aarch64', '24.04'],
        default=True,
    ),
]