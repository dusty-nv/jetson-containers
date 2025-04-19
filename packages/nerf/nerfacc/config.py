from jetson_containers import CUDA_ARCHITECTURES

def nerfacc(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'nerfacc:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'NERFACC_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'nerfacc:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'nerfacc'
        builder['alias'] = 'nerfacc:builder'

    return pkg, builder

package = [
    nerfacc('0.5.3'),
    nerfacc('0.5.4', default=True)
]
