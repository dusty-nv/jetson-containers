from jetson_containers import CUDA_VERSION, SYSTEM_ARM
from packaging.version import Version

def kai_scheduler(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if not version_spec:
        version_spec = version

    pkg['name'] = f'kai_scheduler:{version}'

    pkg['build_args'] = {
        'KAI_SCHEDULER_VERSION': version,
        'KAI_SCHEDULER_VERSION_SPEC': version_spec,
        'SYSTEM_ARM': SYSTEM_ARM
    }

    builder = pkg.copy()

    builder['name'] = f'kai_scheduler:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'kai_scheduler'
        builder['alias'] = 'kai_scheduler:builder'

    return pkg, builder

package = [
    kai_scheduler('0.9.6', '0.9.6', default=True),
]
