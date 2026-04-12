import os
from packaging.version import Version

from jetson_containers import (
    L4T_VERSION, CUDA_VERSION, IS_TEGRA, IS_SBSA, CUDA_ARCH, LSB_RELEASE,
    update_dependencies, package_requires
)

# cudastack consolidates multiple CUDA libraries into ONE RUN to avoid layer limits
# It depends on 'cuda' being already installed

IS_CONFIG = 'package' in globals()  # cuda_stack_args() gets imported by other packages

def cuda_stack_args():
    """
    Get build args for cudastack based on platform and versions
    """
    distro = f"ubuntu{LSB_RELEASE.replace('.', '')}"
    tensorrt_base_url = 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt'

    # Determine component versions based on L4T/CUDA version
    if IS_TEGRA:
        if L4T_VERSION.major >= 39:  # JetPack 6 on Orin
            cudnn_ver = '9.21.0'
            cudnn_url = f"https://developer.download.nvidia.com/compute/cudnn/9.21.0/local_installers/cudnn-local-repo-{distro}-9.21.0_1.0-1_arm64.deb"
            cudnn_packages = "libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-samples"
            tensorrt_ver = '10.16.0'
            tensorrt_url = f"{tensorrt_base_url}/10.16.0/tars/TensorRT-10.16.0.72.Linux.aarch64-gnu.cuda-13.2.tar.gz"
            nccl_ver = '2.29.7'
            nccl_url = f"https://developer.download.nvidia.com/compute/redist/nccl/v{nccl_ver}/nccl_{nccl_ver}-1+cuda13.2_aarch64.txz"
        elif L4T_VERSION.major >= 36:  # JetPack 6
            if CUDA_VERSION >= Version('13.2'):
                cudnn_ver = '9.21.0'
                cudnn_url = f"https://developer.download.nvidia.com/compute/cudnn/9.21.0/local_installers/cudnn-local-repo-{distro}-9.21.0_1.0-1_arm64.deb"
                cudnn_packages = "libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-samples"
                tensorrt_ver = '10.16.0'
                tensorrt_url = f"{tensorrt_base_url}/10.16.0/tars/TensorRT-10.16.0.72.Linux.aarch64-gnu.cuda-13.2.tar.gz"
                nccl_ver = '2.29.7'
            elif CUDA_VERSION >= Version('12.9'):
                cudnn_ver = '9.15.0'
                cudnn_url = f"https://developer.download.nvidia.com/compute/cudnn/9.15.0/local_installers/cudnn-local-tegra-repo-{distro}-9.15.0_1.0-1_arm64.deb"
                cudnn_packages = "libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"
                tensorrt_ver = '10.13'
                tensorrt_url = f"{tensorrt_base_url}/10.13.3/tars/TensorRT-10.13.3.9.Linux.aarch64-gnu.cuda-13.0.tar.gz"
            elif CUDA_VERSION >= Version('12.8'):
                cudnn_ver = '9.8.0'
                cudnn_url = f"https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-tegra-repo-{distro}-9.8.0_1.0-1_arm64.deb"
                cudnn_packages = "libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"
                tensorrt_ver = '10.7'
                tensorrt_url = f"{tensorrt_base_url}/10.7.0/tars/TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz"
            elif CUDA_VERSION >= Version('12.6'):
                cudnn_ver = '9.3.0'
                cudnn_url = f"https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-tegra-repo-{distro}-9.3.0_1.0-1_arm64.deb"
                cudnn_packages = "libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"
                tensorrt_ver = '10.3'
                tensorrt_url = f"{tensorrt_base_url}/10.3.0/tars/TensorRT-10.3.0.26.l4t.aarch64-gnu.cuda-12.6.tar.gz"
            else:  # 12.4
                cudnn_ver = '9.0.0'
                cudnn_url = f"https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-tegra-repo-{distro}-9.0.0_1.0-1_arm64.deb"
                cudnn_packages = "libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples"
                tensorrt_ver = '10.0'
                tensorrt_url = f"{tensorrt_base_url}/10.0.1/tars/TensorRT-10.0.1.6.l4t.aarch64-gnu.cuda-12.4.tar.gz"
            nccl_ver = '2.27.7'
            nccl_url = ''
        else:  # JetPack 5
            cudnn_ver = '8.6.0'
            cudnn_url = "https://repo.download.nvidia.com/jetson/common/pool/main/c/cudnn/libcudnn8_8.6.0.166-1+cuda11.4_arm64.deb"
            cudnn_packages = "libcudnn8 libcudnn8-dev"
            tensorrt_ver = '8.6.0'
            tensorrt_url = "https://nvidia.box.com/shared/static/hmwr57hm88bxqrycvlyma34c3k4c53t9.deb"
            nccl_ver = '2.21.5'
            nccl_url = ''
    elif IS_SBSA:
        cudnn_ver = '9.21.0'
        cudnn_url = f"https://developer.download.nvidia.com/compute/cudnn/9.21.0/local_installers/cudnn-local-repo-{distro}-9.21.0_1.0-1_arm64.deb"
        cudnn_packages = "libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-samples"
        tensorrt_ver = '10.16.0'
        tensorrt_url = f"{tensorrt_base_url}/10.16.0/tars/TensorRT-10.16.0.72.Linux.aarch64-gnu.cuda-13.2.tar.gz"
        nccl_ver = '2.29.7'
        nccl_url = f"https://developer.download.nvidia.com/compute/redist/nccl/v{nccl_ver}/nccl_{nccl_ver}-1+cuda13.2_aarch64.txz"

    else:  # x86_64
        cudnn_ver = '9.21.0'
        cudnn_url = f"https://developer.download.nvidia.com/compute/cudnn/9.21.0/local_installers/cudnn-local-repo-{distro}-9.21.0_1.0-1_amd64.deb"
        cudnn_packages = "libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-samples"
        tensorrt_ver = '10.16.0'
        tensorrt_url = f"{tensorrt_base_url}/10.16.0/tars/TensorRT-10.16.0.72.Linux.x86_64-gnu.cuda-13.2.tar.gz"

        nccl_ver = '2.29.7'
        nccl_url = f"https://developer.download.nvidia.com/compute/redist/nccl/v{nccl_ver}/nccl_{nccl_ver}-1+cuda13.2_x86_64.txz"

    # Extract DEB name from URL
    cudnn_deb = os.path.basename(cudnn_url).split('_')[0] if cudnn_url else ""

    return {
        # cuDNN
        'CUDNN_VERSION': cudnn_ver,
        'CUDNN_URL': cudnn_url,
        'CUDNN_DEB': cudnn_deb,
        'CUDNN_PACKAGES': cudnn_packages,

        # TensorRT
        'TENSORRT_VERSION': tensorrt_ver,
        'TENSORRT_URL': tensorrt_url,
        'TENSORRT_DEB': os.path.basename(tensorrt_url).split('_')[0] if tensorrt_url.endswith('.deb') else '',
        'TENSORRT_PACKAGES': 'tensorrt tensorrt-libs python3-libnvinfer-dev' if tensorrt_url.endswith('.deb') else '',

        # NCCL
        'NCCL_VERSION': nccl_ver,
        'NCCL_URL': nccl_url,

        # Additional libraries
        'CUDSS_VERSION': '0.7.1',
        'CUSPARSELT_VERSION': '0.9.0',
        'CUTENSOR_VERSION': '2.6.0',
        'GDRCOPY_VERSION': '2.5.2',
        'NVPL_VERSION': '25.11',
        'NVSHMEM_VERSION': '3.6.5',

        # Architecture and CUDA info
        'CUDA_ARCH': CUDA_ARCH,
        'CUDA_VERSION_MAJOR': str(CUDA_VERSION.major),
        'CUDA_INSTALLED_VERSION': int(str(CUDA_VERSION.major) + str(CUDA_VERSION.minor)),
        'L4T_VERSION_MAJOR': str(L4T_VERSION.major),
        'DISTRO': distro,
        'IS_SBSA': '1' if IS_SBSA else '0',
        'IS_TEGRA': '1' if IS_TEGRA else '0',
    }


# Export CUDNN_VERSION and TENSORRT_VERSION as global variables for other packages to import
# Convert to Version objects for compatibility with other packages
CUDNN_VERSION = Version(cuda_stack_args()['CUDNN_VERSION'])
TENSORRT_VERSION = Version(cuda_stack_args()['TENSORRT_VERSION'])


def cuda_stack(name, with_tensorrt=False, with_cutlass=False, minimal=False, requires=None, alias=None):
    """
    Generate a consolidated CUDA stack package that installs multiple libraries in ONE RUN.
    This avoids Docker's layer limits by consolidating cudnn, nccl, tensorrt, etc. into one layer.

    Args:
        name: Package name (e.g., 'cudastack:minimal')
        with_tensorrt: Include TensorRT (default: False)
        with_cutlass: Include CUTLASS (default: False)
        minimal: Only install cuDNN and NCCL (default: False)
        requires: Version requirements
        alias: Package aliases
    """
    pkg = package.copy()
    pkg['name'] = name

    if alias:
        if isinstance(alias, str):
            alias = [alias]
        pkg['alias'] = pkg.get('alias', []) + alias

    # Get base build args
    build_args = cuda_stack_args()

    # Component toggles
    build_args['WITH_CUDNN'] = '1'  # Always include cuDNN
    build_args['WITH_TENSORRT'] = '1' if with_tensorrt else '0'
    build_args['WITH_NCCL'] = '1'    # Always include NCCL (minimal or not)
    build_args['WITH_CUTLASS'] = '1' if with_cutlass else '0'
    build_args['WITH_CUDSS'] = '0' if minimal else '1'
    build_args['WITH_CUSPARSELT'] = '0' if minimal else '1'
    build_args['WITH_CUTENSOR'] = '0' if minimal else '1'
    build_args['WITH_GDRCOPY'] = '0' if minimal else '1'
    build_args['WITH_NVPL'] = '0' if minimal else ('1' if IS_SBSA else '0')
    build_args['WITH_NVSHMEM'] = '0' if minimal else '1'

    # Enable distributed (experimental) NCCL for Jetson if explicitly enabled via environment variable
    if IS_TEGRA and os.environ.get('ENABLE_DISTRIBUTED_JETSON_NCCL', '0') == '1':
        build_args['ENABLE_DISTRIBUTED_JETSON_NCCL'] = '1'

    pkg['build_args'] = build_args
    pkg['depends'] = ['cuda']  # cudastack depends ON cuda, doesn't replace it

    if requires:
        pkg['requires'] = requires

    package_requires(pkg, system_arch='aarch64')  # default to aarch64

    return pkg


# Define package variants
# Following the pattern: return a list of packages for different configurations
if IS_CONFIG:
    variants = [
        # Minimal: CUDA + cuDNN + NCCL
        {
            'name': 'cudastack:minimal',
            'alias': 'cuda-stack:minimal',
            'with_tensorrt': False,
            'with_cutlass': False,
            'minimal': True,
        },
        # Standard: + TensorRT
        {
            'name': 'cudastack:standard',
            'alias': ['cuda-stack:standard', 'cudastack', 'cuda-stack'],
            'with_tensorrt': True,
            'with_cutlass': False,
            'minimal': False,
        },
        # Full: + CUTLASS
        {
            'name': 'cudastack:full',
            'alias': 'cuda-stack:full',
            'with_tensorrt': True,
            'with_cutlass': True,
            'minimal': False,
        },
    ]

    # Set architecture requirements
    if IS_TEGRA:
        requires = '>=35'
    elif IS_SBSA:
        requires = 'aarch64'
    else:
        requires = 'x86_64'

    package_list = []

    for v in variants:
        package_list.append(cuda_stack(
            v['name'],
            with_tensorrt=v['with_tensorrt'],
            with_cutlass=v['with_cutlass'],
            minimal=v['minimal'],
            requires=requires,
            alias=v['alias']
        ))
    
    package = package_list
