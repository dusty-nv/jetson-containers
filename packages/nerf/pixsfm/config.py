from jetson_containers import CUDA_ARCHITECTURES

def pixsfm(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'pixsfm:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'PIXSFM_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'pixsfm:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'pixsfm'
        builder['alias'] = 'pixsfm:builder'

    return pkg, builder

package = [
    pixsfm('1.0', default=True)
]