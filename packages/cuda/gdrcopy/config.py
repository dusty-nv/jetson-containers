from jetson_containers import PYTHON_VERSION, LSB_RELEASE
from packaging.version import Version

package['build_args'] = {
    'GDRCOPY_VERSION': 'main',
    'DISTRO': f"ubuntu{LSB_RELEASE.replace('.','')}",
}
