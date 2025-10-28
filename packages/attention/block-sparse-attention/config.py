
from jetson_containers import CUDA_VERSION
from packaging.version import Version

def block_sparse_attn(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'block_sparse_attn:{version}'

    pkg['build_args'] = {
        'BLOCKSPARSEATTN_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'block_sparse_attn:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'block_sparse_attn'
        builder['alias'] = 'block_sparse_attn:builder'

    return pkg, builder

package = [
    block_sparse_attn('0.0.2', default=(CUDA_VERSION >= Version('12.6'))),
]

