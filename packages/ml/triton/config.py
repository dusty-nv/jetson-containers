from ..pytorch.version import PYTORCH_VERSION
from packaging.version import Version

def triton(version, branch=None, requires=None, default=False):
    pkg = package.copy()

    if not branch:
        branch = f'v{version}'
        
    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'triton:{version}'
    
    pkg['build_args'] = {
        'TRITON_VERSION': version,
        'TRITON_BRANCH': branch,
    }
    
    builder = pkg.copy()
    builder['name'] += '-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}
    
    if default:
        pkg['alias'] = 'triton'
        builder['alias'] = 'triton:builder'
        
    return pkg, builder

package = [
    triton('3.4.0', branch='release/3.4.x', default=(PYTORCH_VERSION > Version('2.7'))), # Newer Kernels and Spark Support
    triton('3.3.0', branch='release/3.3.x', default=(PYTORCH_VERSION == Version('2.7'))), # Blackwell/RTX50 Support
    triton('3.2.0', branch='release/3.2.x', default=(PYTORCH_VERSION < Version('2.7'))),
    triton('3.1.0', branch='release/3.1.x'),
    triton('3.0.0', branch='release/3.0.x'),
]

