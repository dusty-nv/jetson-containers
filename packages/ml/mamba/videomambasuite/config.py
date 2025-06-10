from jetson_containers import CUDA_ARCHITECTURES

def videomambasuite(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'videomambasuite:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'VIDEOMAMBASUITE_VERSION': version,
    }

    #builder = pkg.copy()

    #builder['name'] = f'videomamba:{version}-builder'
    #builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'videomambasuite'
        #builder['alias'] = 'videomambasuite:builder'

    return pkg #, builder

package = [
    videomambasuite('1.0', default=True)
]
