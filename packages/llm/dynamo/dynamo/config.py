from jetson_containers import CUDA_VERSION, SYSTEM_ARM
from packaging.version import Version

def dynamo(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    if not version_spec:
        version_spec = version

    pkg['name'] = f'dynamo:{version}'
        
    pkg['build_args'] = {
        'DYNAMO_VERSION': version,
        'DYNAMO_VERSION_SPEC': version_spec,
        'SYSTEM_ARM': SYSTEM_ARM
    }

    builder = pkg.copy()

    builder['name'] = f'dynamo:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'dynamo'
        builder['alias'] = 'dynamo:builder'

    return pkg, builder

package = [
    dynamo('0.2.1', '0.2.1', default=True),
]
