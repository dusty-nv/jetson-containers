from jetson_containers import CUDA_ARCHITECTURES

def tensorflow_addons(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'tensorflow_addons:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TENSORFLOW_ADDONS_VERSION': version,
    }

    #builder = pkg.copy()

    #builder['name'] = f'tensorflow_addons:{version}-builder'
    #builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'tensorflow_addons'
        #builder['alias'] = 'tensorflow_addons:builder'

    return pkg #, builder

package = [
    tensorflow_addons('0.24', default=True)
]
