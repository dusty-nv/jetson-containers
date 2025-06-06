
from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def mem_vid(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'mem_vid:{version}'
    
    pkg['build_args'] = {
        'MEMVID_VERSION': version,
        'IS_SBSA': IS_SBSA
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'mem_vid:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'mem_vid'
        builder['alias'] = 'mem_vid:builder'
        
    return pkg, builder

package = [
    mem_vid('0.1.4', default=True),
]

