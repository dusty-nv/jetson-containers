from jetson_containers import CUDA_VERSION
from packaging.version import Version

def partpacker(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'partpacker:{version}'

    pkg['build_args'] = {
        'PARTPACKER_VERSION': version,
    }

    builder = pkg.copy()

    if default:
        pkg['alias'] = 'partpacker'

    return pkg

package = [
    partpacker('0.1.0', default=True)
]
