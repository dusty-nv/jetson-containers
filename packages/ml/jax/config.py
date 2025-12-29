from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES, CUDA_VERSION, IS_SBSA

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
        'IS_SBSA': int(IS_SBSA),
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
    # Note: each L4T version requirement must have at least a single default JAX version
    jax('0.4.38', requires='==35.*', default=True), # It works from jetpack 5 11.8 Cuda & 8.6 Cudnn
    jax('0.6.2', requires='==36.*', default=True), # It works from jetpack 5 11.8 Cuda & 8.6 Cudnn
    jax('0.9.0', requires='>=38', default=True), # Blackwell Support
]
