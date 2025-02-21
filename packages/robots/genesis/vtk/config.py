from jetson_containers import CUDA_ARCHITECTURES

def vtk(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'vtk:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'VTK_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'vtk:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'vtk'
        builder['alias'] = 'vtk:builder'

    return pkg, builder

package = [
    vtk('9.3.1', default=True),
    vtk('9.4.1')
]
