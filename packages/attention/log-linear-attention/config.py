
from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def flash_attn(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'flash-attention:{version}'
    
    pkg['build_args'] = {
        'LOG_LINEAR_ATTN_VERSION': version,
        'IS_SBSA': IS_SBSA
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'flash-attention:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'flash-attention'
        builder['alias'] = 'flash-attention:builder'
        
    return pkg, builder

package = [
    flash_attn('2.5.7'),
]

