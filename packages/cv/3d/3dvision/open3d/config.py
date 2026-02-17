from jetson_containers import CUDA_ARCHITECTURES

def open3d(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'open3d:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'OPEN3D_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'open3d:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'open3d'
        builder['alias'] = 'open3d:builder'

    return pkg, builder

package = [
    open3d('0.20.0', default=True)
]
