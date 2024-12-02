from jetson_containers import CUDA_ARCHITECTURES

def sana(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'sana:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'SANA_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'sana:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'sana'
        builder['alias'] = 'sana:builder'

    return pkg, builder

package = [
    sana('1.0', default=True)
]
