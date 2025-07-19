from jetson_containers import PYTHON_VERSION, IS_SBSA
from packaging.version import Version

package['build_args'] = {
    'NCCL_VERSION': '2.27.6',
    'IS_SBSA': IS_SBSA,
}
