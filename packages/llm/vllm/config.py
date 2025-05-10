from jetson_containers import CUDA_VERSION, IS_SBSA
from packaging.version import Version

def vllm(version, branch=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    suffix = branch if branch else version
    branch = branch if branch else f'v{version}'

    pkg['name'] = f'vllm:{suffix}'

    pkg['build_args'] = {
        'VLLM_VERSION': version,
        'VLLM_BRANCH': branch,
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()

    builder['name'] = f'vllm:{suffix}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'vllm'
        builder['alias'] = 'vllm:builder'

    return pkg, builder

package = [
    # 0.6.5 compatible with jetson https://github.com/vllm-project/vllm/pull/9735
    vllm(version='0.7.4', default=False),
    vllm(version='0.8.4', default=False),
    vllm(version='0.8.5', default=False),
    vllm(version='0.8.6', default=True),
]
