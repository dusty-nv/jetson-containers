
from jetson_containers import L4T_VERSION

from packaging.version import Version

# https://www.stereolabs.com/developers/release/
latest_zed_version = Version('35.3.1')

if L4T_VERSION > latest_zed_version:
    L4T_VERSION = latest_zed_version
    
package['build_args'] = {
    'L4T_MAJOR_VERSION': L4T_VERSION.major,
    'L4T_MINOR_VERSION': L4T_VERSION.minor,
    'L4T_PATCH_VERSION': L4T_VERSION.micro,
    'ZED_SDK_MAJOR': 4,
    'ZED_SDK_MINOR': 0,
}
