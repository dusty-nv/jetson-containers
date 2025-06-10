from jetson_containers import PYTHON_VERSION
from packaging.version import Version

package['build_args'] = {
    'GDRCOPY_VERSION': 'main',
}
