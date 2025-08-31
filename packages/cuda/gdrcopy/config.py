from jetson_containers import PYTHON_VERSION, LSB_RELEASE
from packaging.version import Version

package['build_args'] = {
    'GDRCOPY_VERSION': '2.5.1',
    'DISTRO': f"ubuntu{LSB_RELEASE.replace('.','')}",
}
