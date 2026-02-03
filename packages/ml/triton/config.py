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
    triton('3.7.0', branch='release/3.7.x', default=(PYTORCH_VERSION >= Version('2.11'))),
    triton('3.6.0', branch='release/3.6.x', default=(PYTORCH_VERSION >= Version('2.10'))), # Fix Issue DGX Spark
    triton('3.5.1', branch='release/3.5.x', default=(PYTORCH_VERSION >= Version('2.9'))), # Newer Kernels for Thor
    triton('3.4.0', branch='release/3.4.x', default=(PYTORCH_VERSION >= Version('2.8'))), # Newer Kernels and Spark Support
]

