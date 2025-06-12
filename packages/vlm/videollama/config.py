
from jetson_containers import CUDA_VERSION
from packaging.version import Version

def videollama(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'videollama:{version}'
    
    pkg['build_args'] = {
        'VIDEOLLAMA_VERSION': version,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'videollama:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'videollama'
        builder['alias'] = 'videollama:builder'
        
    return pkg, builder

package = [
    videollama('1.0.0', default=(CUDA_VERSION >= Version('12.6'))),
]

