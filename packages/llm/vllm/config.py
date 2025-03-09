
def vllm(vllm_version, xgrammar_version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'vllm:{vllm_version}'

    pkg['build_args'] = {
        'VLLM_VERSION': vllm_version,
        'XGRAMMAR_VERSION': xgrammar_version,
    }

    builder = pkg.copy()

    builder['name'] = f'vllm:{vllm_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'vllm'
        builder['alias'] = 'vllm:builder'

    return pkg, builder

package = [
    # 0.6.5 compatible with jetson https://github.com/vllm-project/vllm/pull/9735
    vllm(vllm_version='0.7.4', xgrammar_version='0.1.15', default=True),
]
