from jetson_containers import CUDA_ARCHITECTURES

def transformer_engine(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'transformer-engine:{version}'

    pkg['build_args'] = {
        'TRANSFORMER_ENGINE_VERSION': version,
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x / 10:.1f}' for x in CUDA_ARCHITECTURES]),
    }

    builder = pkg.copy()

    builder['name'] = f'transformer-engine:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'transformer-engine'
        builder['alias'] = 'transformer-engine:builder'

    return pkg, builder

package = [
    transformer_engine('2.12', default=True) # cutlass 4.2.0 support
]
