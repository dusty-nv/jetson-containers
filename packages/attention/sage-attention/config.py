from jetson_containers import CUDA_VERSION, CUDA_ARCHITECTURES
from packaging.version import Version

def sage_attn(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'sage-attention:{version}'

    pkg['build_args'] = {
        'SAGE_ATTENTION_VERSION': version,
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x / 10:.1f}' for x in CUDA_ARCHITECTURES])
    }

    builder = pkg.copy()

    builder['name'] = f'sage-attention:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'sage-attention'
        builder['alias'] = 'sage-attention:builder'

    return pkg, builder

package = [
    sage_attn('3.0.0', default=(CUDA_VERSION >= Version('12.6'))),
]

