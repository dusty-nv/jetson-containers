from jetson_containers import CUDA_ARCHITECTURES

def causalconv1d(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'causalconv1d:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'CASUALCONV1D_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'causalconv1d:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'causalconv1d'
        builder['alias'] = 'causalconv1d:builder'

    return pkg, builder

package = [
    causalconv1d('1.4.0', default=True)
]
