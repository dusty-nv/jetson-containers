from jetson_containers import CUDA_VERSION
from packaging.version import Version

def bitsandbytes(version, requires=None, default=False, branch=None, repo='bitsandbytes-foundation/bitsandbytes'):
    pkg = package.copy()

    if branch is None:
        branch = version
        
    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'bitsandbytes:{version}'
    
    pkg['build_args'] = {
        'BITSANDBYTES_VERSION': version,
        'BITSANDBYTES_REPO': repo,
        'BITSANDBYTES_BRANCH': branch,
        'CUDA_INSTALLED_VERSION': int(str(CUDA_VERSION.major) + str(CUDA_VERSION.minor)),
        'CUDA_MAKE_LIB': f"cuda{str(CUDA_VERSION.major)}x"
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'bitsandbytes:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'bitsandbytes'
        builder['alias'] = 'bitsandbytes:builder'
        
    return pkg, builder

package = [
    bitsandbytes('0.39.1', default=(CUDA_VERSION < Version('12.2')), repo="dusty-nv/bitsandbytes", branch="main"),
    bitsandbytes('0.44.1', default=(CUDA_VERSION >= Version('12.2'))),
]

