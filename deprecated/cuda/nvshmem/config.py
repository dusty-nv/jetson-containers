from jetson_containers import PYTHON_VERSION, CUDA_ARCH, IS_SBSA, LSB_RELEASE, CUDA_VERSION
from packaging.version import Version

package['build_args'] = {
    'NVSHMEM_VERSION': '3.3.24',
    'CUDA_ARCH': CUDA_ARCH,
    'IS_SBSA': IS_SBSA,
    'DISTRO': f"ubuntu{LSB_RELEASE.replace('.', '')}",
    'CUDA_VERSION_MAJOR': Version(f"{CUDA_VERSION}").major,
}
