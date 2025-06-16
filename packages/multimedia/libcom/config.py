from jetson_containers import CUDA_ARCHITECTURES

def libcom(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'libcom:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'LIBCOM_VERSION': version,
    }

    #builder = pkg.copy()

    #builder['name'] = f'libcom:{version}-builder'
    #builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'libcom'
        #builder['alias'] = 'libcom:builder'

    return pkg #, builder

package = [
    libcom('0.1.0', default=True)
]
