from jetson_containers import CUDA_ARCHITECTURES

def flax(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'flax:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'FLAX_VERSION': version,
    }

    #builder = pkg.copy()

    #builder['name'] = f'flax:{version}-builder'
    #builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'flax'
        #builder['alias'] = 'flax:builder'

    return pkg #, builder

package = [
    flax('0.10.0', default=True)
]
