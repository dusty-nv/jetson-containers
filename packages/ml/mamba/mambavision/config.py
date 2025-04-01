from jetson_containers import CUDA_ARCHITECTURES

def mambavision(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'mambavision:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'MAMBAVISION_VERSION': version,
    }

    #builder = pkg.copy()

    #builder['name'] = f'mambavision:{version}-builder'
    #builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'mambavision'
        #builder['alias'] = 'mambavision:builder'

    return pkg #, builder

package = [
    mambavision('1.0', default=True)
]
