from jetson_containers import CUDA_ARCHITECTURES

def vllm(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'vllm:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
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
    vllm('0.6.0', default=True),
    vllm('0.6.4')
]
