from jetson_containers import CUDA_ARCHITECTURES

def dimba(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'dimba:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'DIMBA_VERSION': version,
    }

    #builder = pkg.copy()

    #builder['name'] = f'dimba:{version}-builder'
    #builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'dimba'
        #builder['alias'] = 'dimba:builder'

    return pkg #, builder

package = [
    dimba('1.0', default=True)
]
