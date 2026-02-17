from jetson_containers import CUDA_VERSION, IS_SBSA, CUDA_ARCHITECTURES
from packaging.version import Version

def usd_core(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'usd_core:{version}'

    pkg['build_args'] = {
        'usd_core_VERSION': version,
        'usd_core_VERSION_SPEC': version_spec if version_spec else version,
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x / 10:.1f}' for x in CUDA_ARCHITECTURES]),
        'IS_SBSA': IS_SBSA,
    }

    builder = pkg.copy()

    builder['name'] = f'usd_core:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'usd_core'
        builder['alias'] = 'usd_core:builder'

    return pkg, builder


package = [
    usd_core('25.05.01', default=(CUDA_VERSION >= Version('12.6'))),
]
