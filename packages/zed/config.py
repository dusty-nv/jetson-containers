
from jetson_containers import L4T_VERSION

package['build_args'] = {
    'L4T_MAJOR_VERSION': L4T_VERSION.major,
    'L4T_MINOR_VERSION': L4T_VERSION.minor,
    'L4T_PATCH_VERSION': L4T_VERSION.micro,
    'ZED_SDK_MAJOR': 4,
    'ZED_SDK_MINOR': 0,
}
