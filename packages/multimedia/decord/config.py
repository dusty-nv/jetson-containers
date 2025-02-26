from jetson_containers import CUDA_ARCHITECTURES, CUDA_VERSION

def decord(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'decord:{version}'

    pkg['build_args'] = {
        'CUDA_VERSION': CUDA_VERSION,
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'DECORD_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'decord:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'decord'
        builder['alias'] = 'decord:builder'

    return pkg, builder

package = [
    decord('0.7.0', default=True)
]
