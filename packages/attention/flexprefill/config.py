from jetson_containers import CUDA_VERSION
from packaging.version import Version

def flex_prefill(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'flexprefill:{version}'

    pkg['build_args'] = {
        'FLEXPREFILL_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'flexprefill:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'flexprefill'
        builder['alias'] = 'flexprefill:builder'

    return pkg, builder

package = [
    flex_prefill('0.1.0', default=(CUDA_VERSION >= Version('12.6'))),
]

