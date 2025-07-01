from jetson_containers import CUDA_ARCHITECTURES

def vtk(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if not version_spec:
        version_spec = version

    pkg['name'] = f'vtk:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'VTK_VERSION': version,
        'VTK_VERSION_SPEC': version_spec,
    }

    builder = pkg.copy()

    builder['name'] = f'vtk:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'vtk'
        builder['alias'] = 'vtk:builder'

    return pkg, builder

package = [
    vtk('9.3.1', version_spec='9.3.1.dev0'),
    vtk('9.4.2', version_spec='9.4.2'),
    vtk('9.5.0', version_spec='9.5.0', default=True), # new aarch64 wheels
]
