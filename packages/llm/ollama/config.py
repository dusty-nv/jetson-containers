import os

from jetson_containers import JETPACK_VERSION, nv_tegra_release, IS_SBSA, CUDA_VERSION

PKG_DIR = os.path.dirname(__file__)
nv_tegra_release(dst=os.path.join(PKG_DIR, 'nv_tegra_release'))

def ollama(version, default=False):
    pkg = package.copy()

    pkg['name'] = f'ollama:{version}'

    pkg['build_args'] = {
        'OLLAMA_VERSION': version,
        'JETPACK_VERSION_MAJOR': JETPACK_VERSION.major,
        'IS_SBSA': IS_SBSA,
        'CUDA_VERSION_MAJOR': CUDA_VERSION.major,
    }

    if default:
        pkg['alias'] = 'ollama'

    return pkg

package = [
    #ollama('main'),
    ollama('0.15.4', default=True)
]
