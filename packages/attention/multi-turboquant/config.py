from jetson_containers import CUDA_VERSION
from packaging.version import Version


def multi_turboquant(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'multi-turboquant:{version}'

    pkg['build_args'] = {
        'MULTI_TURBOQUANT_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'multi-turboquant:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'multi-turboquant'
        builder['alias'] = 'multi-turboquant:builder'

    return pkg, builder


package = [
    multi_turboquant('0.1.0', default=(CUDA_VERSION >= Version('12.6'))),
]
