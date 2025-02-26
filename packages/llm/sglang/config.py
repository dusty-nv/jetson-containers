from jetson_containers import CUDA_ARCHITECTURES, CUDA_VERSION

def sglang(sglang_version, flashinfer_version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'sglang:{sglang_version}'

    pkg['build_args'] = {
        'CUDA_VERSION': CUDA_VERSION,
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'SGLANG_VERSION': sglang_version,
        'FLASHINFER_VERSION': flashinfer_version,
    }

    builder = pkg.copy()

    builder['name'] = f'sglang:{sglang_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'sglang'
        builder['alias'] = 'sglang:builder'

    return pkg, builder

package = [
    sglang(sglang_version='0.4.4', flashinfer_version='0.2.3', default=True),
]
