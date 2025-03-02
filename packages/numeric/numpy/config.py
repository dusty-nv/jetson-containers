
from jetson_containers import CUDA_VERSION
from packaging.version import Version

NUMPY_PACKAGE = 'numpy<2' \
  if CUDA_VERSION <= Version('12.6') \
    else 'numpy'

package['build_args'] = {
  'NUMPY_PACKAGE': NUMPY_PACKAGE
}