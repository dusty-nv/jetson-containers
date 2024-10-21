from jetson_containers import CUDA_ARCHITECTURES

def deepspeed(version, branch=None, default=False):
    pkg = package.copy()
    
    if not branch:
        branch = f'v{version}'
        
    pkg['name'] = f'deepspeed:{version}'
    
    pkg['build_args'] = {
        'DEEPSPEED_VERSION': version,
        'DEEPSPEED_BRANCH': branch
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'deepspeed:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'deepspeed'
        builder['alias'] = 'deepspeed:builder'

    return pkg, builder

package = [
    deepspeed('0.15.2', default=True),
]
