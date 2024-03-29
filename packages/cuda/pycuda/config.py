from jetson_containers import PYTHON_VERSION
from packaging.version import Version

package['build_args'] = {
    # v2022.1 is the last version to support Python 3.6
    'PYCUDA_VERSION': 'v2022.1' if PYTHON_VERSION == Version('3.6') else 'main',
}
