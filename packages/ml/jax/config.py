from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES, CUDA_VERSION

def jax(version, requires=None, alias=None, default=False):
    """
    Install JAX from pip server or build the wheel from source.
    """
    pkg = package.copy()

    pkg['name'] = f'jax:{version}'
    pkg['dockerfile'] = 'Dockerfile'

    if len(version.split('.')) < 3:
        build_version = version + '.0'
    else:
        build_version = version

    pkg['build_args'] = {
        'JAX_CUDA_ARCH_ARGS': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]),
        'JAX_VERSION': version,
        'JAX_BUILD_VERSION': build_version,
        'CUDA_VERSION': CUDA_VERSION,

    }

    if L4T_VERSION.major >= 36:
        pkg['build_args']['ENABLE_NCCL'] = 1

    if requires:
        pkg['requires'] = requires


    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if default:
        pkg['alias'] = 'jax'
        builder['alias'] = 'jax:builder'

    return pkg, builder


package = [
    jax('0.4.38', requires='>=35'), # It works from jetpack 5 11.8 Cuda & 8.6 Cudnn
    jax('0.6.2', requires='==36.*'), # It works from jetpack 5 11.8 Cuda & 8.6 Cudnn
    jax('0.7.2', requires='>=38', default=True), # Blackwell Support
]
