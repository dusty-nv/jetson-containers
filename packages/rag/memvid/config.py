
from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def memvid(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'memvid:{version}'
    
    pkg['build_args'] = {
        'MEMVID_VERSION': version,
        'IS_SBSA': IS_SBSA
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'memvid:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'memvid'
        builder['alias'] = 'memvid:builder'
        
    return pkg, builder

package = [
    memvid('0.1.4', default=True),
]

