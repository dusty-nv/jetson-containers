from jetson_containers import CUDA_ARCHITECTURES

def cobra(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'cobra:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'COBRA_VERSION': version,
    }

    #builder = pkg.copy()

    #builder['name'] = f'cobra:{version}-builder'
    #builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'cobra'
        #builder['alias'] = 'cobra:builder'

    return pkg #, builder

package = [
    cobra('0.0.1', default=True)
]
