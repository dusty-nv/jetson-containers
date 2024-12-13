from jetson_containers import CUDA_ARCHITECTURES

def pycolmap(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'pycolmap:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'PYCOLMAP_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'pycolmap:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'pycolmap'
        builder['alias'] = 'pycolmap:builder'

    return pkg, builder

package = [
    pycolmap('3.8'),
    pycolmap('3.10'),
    pycolmap('3.11', default=True),
]
