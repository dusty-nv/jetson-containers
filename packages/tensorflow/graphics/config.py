from jetson_containers import CUDA_ARCHITECTURES

def tensorflow_graphics(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'tensorflow_graphics:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TENSORFLOW_GRAPHICS_VERSION': version,
    }

    #builder = pkg.copy()

    #builder['name'] = f'tensorflow_graphics:{version}-builder'
    #builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'tensorflow_graphics'
        #builder['alias'] = 'tensorflow_graphics:builder'

    return pkg #, builder

package = [
    tensorflow_graphics('2.0', default=True)
]
