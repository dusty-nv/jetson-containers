from jetson_containers import CUDA_VERSION
from packaging.version import Version

def sparc3d(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'sparc3d:{version}'

    pkg['build_args'] = {
        'SPARC3D_VERSION': version,
    }

    builder = pkg.copy()

    if default:
        pkg['alias'] = 'sparc3d'

    return pkg

package = [
    sparc3d('0.1.0', default=True)
]
