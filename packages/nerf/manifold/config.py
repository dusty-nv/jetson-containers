from jetson_containers import CUDA_ARCHITECTURES

def manifold(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'manifold:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'MANIFOLD_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'manifold:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'manifold'
        builder['alias'] = 'manifold:builder'

    return pkg, builder

package = [
    manifold('2.5.1'),
    manifold('2.5.2', default=True),
]
