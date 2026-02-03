from jetson_containers import update_dependencies
from packaging.version import Version
from ..pytorch.version import PYTORCH_VERSION


def torchcodec(version, pytorch=None, depends=None, requires=None):
    pkg = package.copy()

    if pytorch:
        pkg['depends'] = update_dependencies(pkg['depends'], f"pytorch:{pytorch}")
    else:
        pytorch = PYTORCH_VERSION
    # Add pytorch version to package name for additional uniqueness.
    pkg['name'] = f"torchcodec:{version.split('-')[0]}-pytorch:{pytorch}"  # remove any -rc* suffix

    if requires:
        pkg['requires'] = requires

    if depends:
        pkg['depends'] = update_dependencies(pkg['depends'], depends)

    if not isinstance(pytorch, Version):
        pytorch = Version(pytorch)

    if len(version.split('.')) < 3:
        version = version + '.0'

    pkg['build_args'] = {
        'TORCHCODEC_VERSION': version,
    }

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if pytorch == PYTORCH_VERSION:
        pkg['alias'] = 'torchcodec'
        builder['alias'] = 'torchcodec:builder'

    return pkg, builder


package = [
    # JetPack 5/6/7
    torchcodec('0.9.1', pytorch='2.9.1', depends=['ffmpeg:8.0'], requires='>=36'),
    torchcodec('0.10.0', pytorch='2.10', depends=['ffmpeg:8.0'], requires='>=36'),
    torchcodec('0.11.0', pytorch='2.11', depends=['ffmpeg:8.0'], requires='>=36'),
]
