
def vllm(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'vllm:{version}'

    pkg['build_args'] = {
        'VLLM_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'vllm:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'vllm'
        builder['alias'] = 'vllm:builder'

    return pkg, builder

package = [
    # 0.6.5 compatible with jetson https://github.com/vllm-project/vllm/pull/9735
    vllm('0.6.7', default=True),
]
