
from jetson_containers import CUDA_VERSION
from packaging.version import Version

def paraattention(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'paraattention:{version}'
    
    pkg['build_args'] = {
        'PARAATENTTION_VERSION': version,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'paraattention:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'paraattention'
        builder['alias'] = 'paraattention:builder'
        
    return pkg, builder

package = [
    paraattention('0.4.0', default=(CUDA_VERSION >= Version('12.6'))),
]

