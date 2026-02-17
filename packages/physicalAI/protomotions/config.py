from jetson_containers import CUDA_ARCHITECTURES

def protomotions(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'protomotions:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'PROTOMOTIONS_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'protomotions:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'protomotions'
        builder['alias'] = 'protomotions:builder'

    return pkg, builder

package = [
    protomotions('2.5.0', default=True)
]
