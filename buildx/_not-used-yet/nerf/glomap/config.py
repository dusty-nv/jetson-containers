from jetson_containers import CUDA_ARCHITECTURES

def glomap(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'glomap:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'GLOMAP_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'glomap:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'glomap'
        builder['alias'] = 'glomap:builder'

    return pkg, builder

package = [
    glomap('2.0.0', default=True)
]
