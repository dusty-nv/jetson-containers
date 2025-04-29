
def vllm(vllm_version, xgrammar_version, branch=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    suffix = branch if branch else vllm_version
    branch = branch if branch else f'v{vllm_version}'

    pkg['name'] = f'vllm:{suffix}'

    pkg['build_args'] = {
        'VLLM_VERSION': vllm_version,
        'VLLM_BRANCH': branch,
        'XGRAMMAR_VERSION': xgrammar_version,
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
    vllm(vllm_version='0.7.4', xgrammar_version='0.1.15', default=False),
    vllm(vllm_version='0.8.4', xgrammar_version='0.1.18', default=False),
    vllm(vllm_version='0.8.5', xgrammar_version='0.1.18', default=False),
    vllm(vllm_version='0.8.6', xgrammar_version='0.1.19', default=True),
]
