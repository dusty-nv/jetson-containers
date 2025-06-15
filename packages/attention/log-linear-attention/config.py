
from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def log_linear_attn(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'log-linear-attention:{version}'
    
    pkg['build_args'] = {
        'LOG_LINEAR_ATTN_VERSION': version,
        'IS_SBSA': IS_SBSA
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'log-linear-attention:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'log-linear-attention'
        builder['alias'] = 'log-linear-attention:builder'
        
    return pkg, builder

package = [
    log_linear_attn('0.0.1', default=True),
]

