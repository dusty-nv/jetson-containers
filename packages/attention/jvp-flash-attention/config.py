from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def jvp_flash_attn(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'jvp-flash-attention:{version}'

    pkg['build_args'] = {
        'JVP_FLASH_ATTENTION_VERSION': version,
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()

    builder['name'] = f'jvp-flash-attention:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'jvp-flash-attention'
        builder['alias'] = 'jvp-flash-attention:builder'

    return pkg, builder

package = [
    jvp_flash_attn('0.10.0', default=(CUDA_VERSION >= Version('12.6'))),
]

