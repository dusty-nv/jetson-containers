from jetson_containers import CUDA_ARCHITECTURES

def fruitnerf(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'fruitnerf:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'FRUITNERF_VERSION': version,
    }

    #builder = pkg.copy()

    #builder['name'] = f'fruitnerf:{version}-builder'
    #builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'fruitnerf'
        #builder['alias'] = 'fruitnerf:builder'

    return pkg #, builder

package = [
    fruitnerf('1.0', default=True)
]
