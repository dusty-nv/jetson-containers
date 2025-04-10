from jetson_containers import CUDA_ARCHITECTURES

def deepspeed_kernels(version, branch=None, default=False):
    pkg = package.copy()
    
    if not branch:
        branch = f'v{version}'
        
    pkg['name'] = f'deepspeed-kernels:{version}'
    
    pkg['build_args'] = {
        'DEEPSPEED_KERNELS_VERSION': version,
        'DEEPSPEED_KERNELS_BRANCH': branch
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'deepspeed-kernels:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'deepspeed-kernels'
        builder['alias'] = 'deepspeed-kernels:builder'

    return pkg, builder

package = [
    deepspeed_kernels('0.1.0', branch='main', default=True),
]
