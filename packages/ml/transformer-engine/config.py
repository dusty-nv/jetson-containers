from jetson_containers import CUDA_ARCHITECTURES

def transformer_engine(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'transformer-engine:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TRANSFORMER_ENGINE_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'transformer-engine:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'transformer-engine'
        builder['alias'] = 'transformer-engine:builder'

    return pkg, builder

package = [
    transformer_engine('1.13'),
    transformer_engine('2.0'), # Blackwell support
    transformer_engine('2.1'), # FP4 support
    transformer_engine('2.2', default=True) # cutlass 3.9.0 support
]
