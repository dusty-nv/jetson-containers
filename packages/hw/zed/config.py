
from jetson_containers import L4T_VERSION, CUDA_VERSION, LSB_RELEASE, IS_TEGRA
from packaging.version import Version

from ..robots.ros import ros_container

def zed(version, l4t_version=None, requires=None, default=False):
    """
    Container for Stereolabs ZED SDK from:

      https://www.stereolabs.com/developers/release
      https://github.com/stereolabs/zed-docker

    This defines a version released to Stereolabs website.
    """
    sdk = package.copy()

    if l4t_version and IS_TEGRA:
        url = f"https://download.stereolabs.com/zedsdk/{version}/l4t{l4t_version}/jetsons"
    else:
        url = f"https://download.stereolabs.com/zedsdk/{version}/cu{CUDA_VERSION.major}/ubuntu{LSB_RELEASE.split('.')[0]}"
    
    sdk['build_args'] = {
        'ZED_URL': url,
        'L4T_MAJOR_VERSION': L4T_VERSION.major,
        'L4T_MINOR_VERSION': L4T_VERSION.minor,
        'L4T_PATCH_VERSION': L4T_VERSION.micro,
    }

    sdk['name'] = f'zed:{version}'

    if default:
        sdk['alias'] = 'zed'

    if requires:
        sdk['requires'] = requires

    return [
        sdk,
        ros_container({
            **sdk,
            'name': sdk['name'] + '-${ROS_DISTRO}',
            'alias': sdk['alias'] + '-${ROS_DISTRO}',
            }, 
            'https://github.com/stereolabs/zed-ros2-interfaces',
            'https://github.com/stereolabs/zed-ros2-wrapper',
            distros=['humble', 'jazzy'], base_packages='desktop'
        )
    ]

package = [
    zed('5.0', '35.4', requires='<=35', default=True),  # JetPack 5
    zed('5.0', '36.4', requires='>=36', default=True),  # JetPack 6
]
