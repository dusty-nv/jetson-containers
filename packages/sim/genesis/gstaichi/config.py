from jetson_containers import CUDA_ARCHITECTURES

def gstaichi(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'gstaichi:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'GSTAICHI_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'gstaichi:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'gstaichi'
        builder['alias'] = 'gstaichi:builder'

    return pkg, builder

package = [
    gstaichi('4.7.0', default=True)
]
