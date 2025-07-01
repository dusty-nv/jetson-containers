from jetson_containers import CUDA_VERSION, IS_SBSA, update_dependencies
from packaging.version import Version

def vllm(version, branch=None, requires=None, default=False, depends=None):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if depends:
        pkg['depends'] = update_dependencies(pkg['depends'], depends)

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
    vllm(version='0.8.4', depends=['flashinfer:0.2.1.post2'], default=False),
    vllm(version='0.8.5', branch='v0.8.5.post1', depends=['flashinfer:0.2.2.post1'], default=False),
    vllm(version='0.9.0', depends=['flashinfer'], default=False),
    vllm(version='0.9.2', depends=['flashinfer'], default=False),
    vllm(version='0.9.3', depends=['flashinfer'], default=True),
]
