
from jetson_containers import CUDA_VERSION
from packaging.version import Version

def sparge_attn(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'sparge-attention:{version}'
    
    pkg['build_args'] = {
        'SPARGE_ATTENTION_VERSION': version,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'sparge-attention:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'sparge-attention'
        builder['alias'] = 'sparge-attention:builder'
        
    return pkg, builder

package = [
    sparge_attn('0.1.0', default=(CUDA_VERSION >= Version('12.6'))),
]

