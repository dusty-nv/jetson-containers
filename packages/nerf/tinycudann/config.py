from jetson_containers import CUDA_ARCHITECTURES

def tinycudann(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'tinycudann:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TINYCUDANN_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'tinycudann:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'tinycudann'
        builder['alias'] = 'tinycudann:builder'

    return pkg, builder

package = [
    tinycudann('1.7', default=True),
]