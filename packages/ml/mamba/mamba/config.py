from jetson_containers import CUDA_ARCHITECTURES

def mamba(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'mamba:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'MAMBA_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'mamba:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'mamba'
        builder['alias'] = 'mamba:builder'

    return pkg, builder

package = [
    mamba('2.2.5', default=True)
]
