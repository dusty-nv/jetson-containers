from jetson_containers import CUDA_VERSION, SYSTEM_ARM, CUDA_ARCHITECTURES
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
        'COMPUTE_CAPABILITIES': ','.join([str(x) for x in CUDA_ARCHITECTURES]),
        'CUDA_COMPUTE_CAP': ' '.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x / 10:.1f}' for x in CUDA_ARCHITECTURES]),
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
    dynamo('0.8.1', '0.8.1', default=True),
]
