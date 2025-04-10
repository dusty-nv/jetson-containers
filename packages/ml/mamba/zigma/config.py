from jetson_containers import CUDA_ARCHITECTURES

def zigma(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'zigma:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'ZIGMA_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'zigma:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'zigma'
        builder['alias'] = 'zigma:builder'

    return pkg, builder

package = [
    zigma('1.0', default=True)
]
