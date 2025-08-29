from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

if CUDA_VERSION < Version('12.6'):
    NUMPY_PACKAGE = 'numpy<2'
    NUMPY_VERSION_MAJOR = 1
else:
    NUMPY_PACKAGE = 'numpy'
    NUMPY_VERSION_MAJOR = 2

if IS_SBSA:
    ARCH_OPT = "NEOVERSEV2"
else:
    ARCH_OPT = "ARMV8"

package['build_args'] = {
    'NUMPY_PACKAGE': NUMPY_PACKAGE,
    'NUMPY_VERSION_MAJOR': NUMPY_VERSION_MAJOR,
    'IS_SBSA': IS_SBSA,
    'ARCH_OPT': ARCH_OPT,
}
