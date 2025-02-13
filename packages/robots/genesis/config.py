from jetson_containers import CUDA_ARCHITECTURES

def genesis(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'genesis:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'GENESIS_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'genesis:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'genesis'
        builder['alias'] = 'genesis:builder'

    return pkg, builder

package = [
    genesis('0.2.2', default=True)
]
