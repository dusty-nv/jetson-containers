from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def radial_attn(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'radial-attention:{version}'

    pkg['build_args'] = {
        'FLASH_ATTENTION_VERSION': version,
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()

    builder['name'] = f'radial-attention:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'radial-attention'
        builder['alias'] = 'radial-attention:builder'

    return pkg, builder

package = [
    radial_attn('0.1.0', default=(CUDA_VERSION >= Version('12.6'))),
]

