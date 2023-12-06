
from jetson_containers import PYTHON_VERSION
from packaging.version import Version

if PYTHON_VERSION == Version('3.6'):
    PYCUDA_VERSION = 'v2022.1'  # last version to support Python 3.6
else:
    PYCUDA_VERSION = 'main'

package['build_args'] = {
    'PYCUDA_VERSION': PYCUDA_VERSION,
}
