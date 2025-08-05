from jetson_containers import PYTHON_VERSION, IS_SBSA, LSB_RELEASE, CUDA_ARCH
from packaging.version import Version

package['build_args'] = {
    'NCCL_VERSION': '2.27.7',
    'IS_SBSA': IS_SBSA,
    'CUDA_ARCH': CUDA_ARCH,
    'DISTRO': f"ubuntu{LSB_RELEASE.replace('.', '')}",
}
