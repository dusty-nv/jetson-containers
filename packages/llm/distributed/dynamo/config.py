from jetson_containers import CUDA_VERSION, IS_SBSA
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
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()

    builder['name'] = f'dynamo:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'dynamo'
        builder['alias'] = 'dynamo:builder'

    return pkg, builder

package = [
    dynamo('0.3.0', '0.2.0', default=True),
]