
from jetson_containers import CUDA_VERSION
from packaging.version import Version

def sage_attn(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'sage-attention:{version}'
    
    pkg['build_args'] = {
        'SAGE_ATTENTION_VERSION': version,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'sage-attention:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'sage-attention'
        builder['alias'] = 'sage-attention:builder'
        
    return pkg, builder

package = [
    sage_attn('1.0.7', default=(CUDA_VERSION >= Version('12.6'))),
]

