from jetson_containers import CUDA_ARCHITECTURES

def vhacdx(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'vhacdx:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'VHACDX_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'vhacdx:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'vhacdx'
        builder['alias'] = 'vhacdx:builder'

    return pkg, builder

package = [
    vhacdx('0.0.9.post1', default=True)
]