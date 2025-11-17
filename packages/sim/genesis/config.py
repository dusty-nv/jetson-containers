from jetson_containers import CUDA_ARCHITECTURES

def genesis(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'genesis-world:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'GENESIS_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'genesis-world:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'genesis-world'
        builder['alias'] = 'genesis-world:builder'

    return pkg, builder

package = [
    genesis('0.3.8', default=True)
]
