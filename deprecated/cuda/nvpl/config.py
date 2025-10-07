from jetson_containers import PYTHON_VERSION, CUDA_ARCH, IS_SBSA, LSB_RELEASE
from packaging.version import Version

package['build_args'] = {
    'NVPL_VERSION': '25.5',
    'CUDA_ARCH': CUDA_ARCH,
    'IS_SBSA': IS_SBSA,
    'DISTRO': f"ubuntu{LSB_RELEASE.replace('.', '')}",
}
