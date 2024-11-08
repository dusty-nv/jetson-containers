from jetson_containers import CUDA_ARCHITECTURES

def hloc(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'hloc:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'HLOC_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'hloc:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'hloc'
        builder['alias'] = 'hloc:builder'

    return pkg, builder

package = [
    hloc('1.4'),
    hloc('1.5', default=True)
]
