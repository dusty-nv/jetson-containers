import os
from packaging.version import Version

from jetson_containers import (
    L4T_VERSION, JETPACK_VERSION, CUDA_VERSION,
    CUDA_ARCHITECTURES, LSB_RELEASE, CUDA_ARCH, IS_SBSA, IS_TEGRA,
    SYSTEM_ARM, DOCKER_ARCH, package_requires
)


def cuda_build_args(version):
    """
    Return some common environment settings used between variants of the CUDA containers.
    """
    return {
        'CUDA_ARCH_LIST': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'CUDA_ARCH': CUDA_ARCH,
        'CUDA_INSTALLED_VERSION': int(str(CUDA_VERSION.major) + str(CUDA_VERSION.minor)),
        'DISTRO': f"ubuntu{LSB_RELEASE.replace('.','')}",
        'IS_SBSA': IS_SBSA,
        'IS_TEGRA': IS_TEGRA,
    }


def cuda_package(version, url, deb=None, packages=None, requires=None) -> list:
    """
    Generate containers for a particular version of CUDA installed from debian packages
    This will download & install the specified packages (by default the full CUDA Toolkit)
    from a .deb URL from developer.nvidia.com/cuda-downloads (the `aarch64-jetson` versions)
    """
    if not deb:
        deb = url.split('/')[-1].split('_')[0]

    if not packages:
        packages = os.environ.get('CUDA_PACKAGES', 'cuda-toolkit*')

    cuda = package.copy()

    cuda['name'] = f'cuda:{version}'

    cuda['build_args'] = {**{
        'CUDA_URL': url,
        'CUDA_DEB': deb,
        'CUDA_PACKAGES': packages,
        'IS_SBSA': IS_SBSA,
    }, **cuda_build_args(version) }

    if requires:
        cuda['requires'] = requires

    package_requires(cuda, system_arch='aarch64') # default to aarch64

    if 'toolkit' in packages or 'dev' in packages:
        cuda['depends'] = ['build-essential']

    if Version(version) == CUDA_VERSION:
        cuda['alias'] = 'cuda'

    cuda_pip = pip_cache(version, requires)
    cuda['depends'].append(cuda_pip['name'])

    return cuda, cuda_pip


def cuda_builtin(version, requires=None) -> list:
    """
    Backwards-compatability for when CUDA already installed in base container (like l4t-jetpack)
    This will just retag the base, marking CUDA dependency as satisfied in any downstream containers.
    """
    passthrough = package.copy()

    if not isinstance(version, str):
        version = f'{version.major}.{version.minor}'

    passthrough['name'] = f'cuda:{version}'

    passthrough['dockerfile'] = 'Dockerfile.builtin'
    passthrough['build_args'] = cuda_build_args(version)

    if Version(version) == CUDA_VERSION:
        passthrough['alias'] = 'cuda'

    if requires:
        passthrough['requires'] = requires

    passthrough['depends'] = ['build-essential']

    cuda_pip = pip_cache(version, requires)
    passthrough['depends'].append(cuda_pip['name'])

    return passthrough, cuda_pip


def cuda_samples(version, requires, branch=None) -> list:
    """
    Generates container that installs/builds the CUDA samples
    """
    samples = package.copy()

    if not isinstance(version, str):
        version = f'{version.major}.{version.minor}'

    samples['name'] = f'cuda:{version}-samples'
    samples['dockerfile'] = 'Dockerfile.samples'
    samples['notes'] = "CUDA samples from https://github.com/NVIDIA/cuda-samples installed under /opt/cuda-samples"
    samples['test'] = 'test-samples.sh'
    samples['depends'] = [f'cuda:{version}', 'cmake']

    if not branch:
        branch = 'v' + version

    if Version(version) > Version('12.5'):
        make_cmd='cmake'
        samples['depends'] += ['opengl', 'vulkan']
    elif Version(version) >= Version('12.0'):
        make_cmd='make'
    else:
        make_cmd='make_flat'

    samples['build_args'] = {
        'CUDA_BRANCH': branch,
        'CUDA_SAMPLES_MAKE': make_cmd
    }

    if Version(version) == CUDA_VERSION:
        samples['alias'] = 'cuda:samples'

    if requires:
        samples['requires'] = requires

    return samples


def pip_cache(version, requires=None):
    """
    Defines a container that just sets the environment for using the pip caching server.
    https://github.com/dusty-nv/jetson-containers/blob/master/docs/build.md#pip-server
    """
    short_version = f"cu{version.replace('.', '')}"
    index_host = os.environ.get('INDEX_HOST', 'jetson-ai-lab.io')

    pip_path = (
        f"sbsa/{short_version}" if IS_SBSA
        else f"jp{JETPACK_VERSION.major}/{short_version}" if IS_TEGRA
        else f"{DOCKER_ARCH}/{short_version}"
    )

    apt_path = pip_path if Version(LSB_RELEASE).major < 24 else f"{pip_path}/{LSB_RELEASE}"

    pip_cache = package.copy()

    pip_cache['name'] = f'pip_cache:{short_version}'
    pip_cache['group'] = 'build'
    pip_cache['dockerfile'] = 'Dockerfile.pip'
    pip_cache['depends'] = []
    pip_cache['test'] = []

    pip_cache['build_args'] = {
        'TAR_INDEX_URL': os.environ.get('LOCAL_TAR_INDEX_URL', f"https://apt.{index_host}") + f"/{apt_path}",
        'FALLBACK_TAR_INDEX_URL': f"https://apt.{index_host}/{apt_path}",
        'MULTIARCH_URL': f"https://apt.{index_host}/multiarch",
        'DOWNLOADS_URL': f"https://apt.{index_host}/assets",
        'PIP_INDEX_REPO': os.environ.get('LOCAL_PIP_INDEX_URL', os.environ.get('LOCAL_PIP_INDEX_REPO', f"https://pypi.{index_host}") + f"/{pip_path}"),
        'PIP_TRUSTED_HOSTS': f"{os.environ.get('PIP_UPLOAD_HOST', index_host)}",
        'UV_EXTRA_INDEX_URL': f"https://pypi.{index_host}" + "/root/pypi/+simple",
        'PIP_UPLOAD_REPO': os.environ.get('PIP_UPLOAD_REPO', f"http://{os.environ.get('PIP_UPLOAD_HOST', 'localhost')}/{pip_path}"),
        'PIP_UPLOAD_USER': os.environ.get('PIP_UPLOAD_USER', f"jp{JETPACK_VERSION.major}" if SYSTEM_ARM else 'amd64'),
        'PIP_UPLOAD_PASS': os.environ.get('PIP_UPLOAD_PASS', 'none'),
        'SCP_UPLOAD_URL': os.environ.get('SCP_UPLOAD_URL', f"{os.environ.get('SCP_UPLOAD_HOST', 'localhost:/dist')}/{apt_path}"),
        'SCP_UPLOAD_USER': os.environ.get('SCP_UPLOAD_USER'),
        'SCP_UPLOAD_PASS': os.environ.get('SCP_UPLOAD_PASS'),
        'SCP_UPLOAD_KEY': os.environ.get('SCP_UPLOAD_KEY'),
    }

    if requires:
        pip_cache['requires'] = requires

    if Version(version) == CUDA_VERSION:
        pip_cache['alias'] = ['pip_cache', 'pip-cache']

    return pip_cache

if IS_TEGRA:
    package = [
        # JetPack 7
        cuda_package('13.1','https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda-repo-ubuntu2404-13-1-local_13.1.0-590.44.01-1_arm64.deb', requires='aarch64'),
        cuda_samples('13.1', requires='aarch64'),
        cuda_package('13.0', 'https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_arm64.deb', requires='>=38'),
        cuda_samples('13.0', requires='>=38'),

        # JetPack 6
        cuda_package('12.2', 'https://nvidia.box.com/shared/static/uvqtun1sc0bq76egarc8wwuh6c23e76e.deb', 'cuda-tegra-repo-ubuntu2204-12-2-local', requires='==36.*'),
        cuda_samples('12.2', requires='==36.*'),
        cuda_package('12.4', 'https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-tegra-repo-ubuntu2204-12-4-local_12.4.1-1_arm64.deb', requires='==36.*'),
        cuda_samples('12.4', requires='==36.*'),
        cuda_package('12.6', 'https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-tegra-repo-ubuntu2204-12-6-local_12.6.3-1_arm64.deb', requires='==36.*'),
        cuda_samples('12.6', branch='12.5', requires='==36.*'),
        cuda_package('12.8', 'https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-tegra-repo-ubuntu2204-12-8-local_12.8.1-1_arm64.deb', requires='>=36'),
        cuda_samples('12.8', requires='>=36'),
        cuda_package('12.9','https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-tegra-repo-ubuntu2204-12-9-local_12.9.1-1_arm64.deb', requires='>=36'),
        cuda_samples('12.9', requires='>=36'),

        # JetPack 5
        cuda_package('12.2', 'https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-tegra-repo-ubuntu2004-12-2-local_12.2.2-1_arm64.deb', requires='==35.*'),
        cuda_samples('12.2', requires='==35.*'),
        cuda_package('11.8', 'https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb', requires='==35.*'),
        cuda_samples('11.8', requires='==35.*'),

        # JetPack 4-5 (CUDA installed in base container)
        cuda_builtin(CUDA_VERSION, requires='<36'),
        cuda_samples(CUDA_VERSION, requires='<36'),
    ]
elif IS_SBSA:
    package = [
        # sbsa
        cuda_package('12.8',
                     'https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_arm64.deb',
                     requires='aarch64'),
        cuda_samples('12.8', requires='aarch64'),
        cuda_package('12.9',
                     'https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.1-575.57.08-1_arm64.deb',
                     requires='aarch64'),
        cuda_samples('12.9', requires='aarch64'),
        cuda_package('13.0',
                     'https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_arm64.deb',
                     requires='aarch64'),
        cuda_samples('13.0', requires='aarch64'),
        cuda_package('13.1',
                     'https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda-repo-ubuntu2404-13-1-local_13.1.0-590.44.01-1_arm64.deb',
                     requires='aarch64'),
        cuda_samples('13.1', requires='aarch64'),

    ]
else:
    package = [
        # x86_64
        cuda_package('12.8',
                     'https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb',
                     requires='x86_64'),
        cuda_samples('12.8', requires='x86_64'),
        cuda_package('12.9',
                     'https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.1-575.57.08-1_amd64.deb',
                     requires='x86_64'),
        cuda_samples('12.9', requires='x86_64'),
        cuda_package('13.0',
                     'https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_amd64.deb',
                     requires='x86_64'),
        cuda_samples('13.0', requires='x86_64'),
        cuda_package('13.1',
                     'https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda-repo-ubuntu2404-13-1-local_13.1.0-590.44.01-1_amd64.deb',
                     requires='x86_64'),
        cuda_samples('13.1', requires='x86_64'),
    ]
