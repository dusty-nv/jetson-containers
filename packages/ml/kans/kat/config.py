from jetson_containers import CUDA_ARCHITECTURES

def kat(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'kat:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'KAT_VERSION': version,
    }

    #builder = pkg.copy()

    #builder['name'] = f'kat:{version}-builder'
    #builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'kat'
        #builder['alias'] = 'kat:builder'

    return pkg #, builder

package = [
    kat('1.0.0', default=True)
]
