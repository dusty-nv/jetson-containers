from jetson_containers import CUDA_VERSION, IS_SBSA, CUDA_ARCHITECTURES
from packaging.version import Version

# Test flashinfer dependency

def flash_infer(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'flashinfer:{version}'

    pkg['build_args'] = {
        'FLASHINFER_VERSION': version,
        'FLASHINFER_VERSION_SPEC': version_spec if version_spec else version,
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x / 10:.1f}' for x in CUDA_ARCHITECTURES]),
        'FLASHINFER_CUDA_ARCH_LIST': ' '.join([f'{x / 10:.1f}' for x in CUDA_ARCHITECTURES]),
        'IS_SBSA': IS_SBSA,
    }

    builder = pkg.copy()

    builder['name'] = f'flashinfer:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'flashinfer'
        builder['alias'] = 'flashinfer:builder'

    return pkg, builder


package = [
    flash_infer('0.6.5', '0.6.5', default=(CUDA_VERSION >= Version('12.6'))), # Compatible with Spark and Thor
    # flash_infer('latest', 'main', default=(CUDA_VERSION >= Version('12.6'))), # Thor compatibility
]
