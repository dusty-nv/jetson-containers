from jetson_containers import CUDA_ARCHITECTURES

def quadrants(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'quadrants:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'QUADRANTS_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'quadrants:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'quadrants'
        builder['alias'] = 'quadrants:builder'

    return pkg, builder

package = [
    quadrants('5.0.0', default=True)
]
