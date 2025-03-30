from jetson_containers import CUDA_ARCHITECTURES

def nerfstudio(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'nerfstudio:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'NERFSTUDIO_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'nerfstudio:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'nerfstudio'
        builder['alias'] = 'nerfstudio:builder'

    return pkg, builder

package = [
    nerfstudio('1.1.7', default=True)
]
