from jetson_containers import CUDA_ARCHITECTURES

def nvidiamodelopt(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'nvidia_modelopt:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'NVIDIA_MODELOPT_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'nvidia_modelopt:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'nvidia_modelopt'
        builder['alias'] = 'nvidia_modelopt:builder'

    return pkg, builder

package = [
    nvidiamodelopt('0.42.0', default=True)
]
