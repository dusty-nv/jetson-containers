from jetson_containers import CUDA_ARCHITECTURES

def decord(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'decord2:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'DECORD_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'decord2:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'decord2'
        builder['alias'] = 'decord2:builder'

    return pkg, builder

package = [
    decord('3.0.0', default=True)
]
