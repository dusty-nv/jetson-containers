from jetson_containers import PYTHON_VERSION, IS_SBSA, LSB_RELEASE, CUDA_ARCH
from packaging.version import Version

package['build_args'] = {
    'GDRCOPY_VERSION': '2.5.1',
    'IS_SBSA': IS_SBSA,
    'CUDA_ARCH': CUDA_ARCH,
    'DISTRO': f"ubuntu{LSB_RELEASE.replace('.', '')}",
}
