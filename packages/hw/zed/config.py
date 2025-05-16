
from jetson_containers import L4T_VERSION
from packaging.version import Version

def zed(version, l4t_version, requires=None, default=False):
    """
    Container for Stereolabs ZED SDK from:

      https://www.stereolabs.com/developers/release
      https://github.com/stereolabs/zed-docker

    This defines a version released to Stereolabs website.
    """
    sdk = package.copy()
    url = f"https://download.stereolabs.com/zedsdk/{version}/l4t{l4t_version}/jetsons"

    sdk['name'] = f'zed:{version}'

    sdk['build_args'] = {
        'ZED_URL': url,
        'L4T_MAJOR_VERSION': L4T_VERSION.major,
        'L4T_MINOR_VERSION': L4T_VERSION.minor,
        'L4T_PATCH_VERSION': L4T_VERSION.micro,
    }

    if default:
        sdk['alias'] = 'zed'

    if requires:
        sdk['requires'] = requires

    return sdk

package = [
    zed('5.0', '35.4', requires='<=35', default=True),  # JetPack 5
    zed('5.0', '36.4', requires='>=36', default=True),  # JetPack 6
]
