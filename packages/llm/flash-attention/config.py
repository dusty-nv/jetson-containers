
from jetson_containers import CUDA_VERSION
from packaging.version import Version

def flash_attn(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'flash-attention:{version}'
    
    pkg['build_args'] = {
        'FLASH_ATTENTION_VERSION': version,
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
    flash_attn('2.6.3'),
    flash_attn('2.7.2.post1'),
    flash_attn('2.7.4.post1', default=(CUDA_VERSION <= Version('12.8'))),
    # flash_attn('2.7.4.post2', default=(CUDA_VERSION >= Version('12.8'))),
]

