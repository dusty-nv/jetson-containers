from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def dynamo_vllm(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    if not version_spec:
        version_spec = version

    pkg['name'] = f'dynamo_vllm:{version}'
     
    if version:
        vllm_ref = version.split('.post')[0]
        
    pkg['build_args'] = {
        'IS_SBSA': IS_SBSA,
        'VLLM_REF': vllm_ref,
        'VLLM_PATCHED_PACKAGE_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'dynamo_vllm:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'dynamo_vllm'
        builder['alias'] = 'dynamo_vllm:builder'

    return pkg, builder

package = [
    dynamo_vllm('0.2.1', default=True),
]