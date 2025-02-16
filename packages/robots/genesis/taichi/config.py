from jetson_containers import CUDA_ARCHITECTURES

def taichi(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'taichi:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TAICHI_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'taichi:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'taichi'
        builder['alias'] = 'taichi:builder'

    return pkg, builder

package = [
    taichi('1.7.4', default=True)
]
