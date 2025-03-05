from jetson_containers import CUDA_ARCHITECTURES

def openexr(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'openexr:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'OPENEXR_VERSION': version,
    }

    #builder = pkg.copy()

    #builder['name'] = f'openexr:{version}-builder'
    #builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'openexr'
        #builder['alias'] = 'openexr:builder'

    return pkg #, builder

package = [
    openexr('3.4.0', default=True)
]
