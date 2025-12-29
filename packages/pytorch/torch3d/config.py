from packaging.version import Version

def pytorch3d(version, version_spec=None, requires=None, depends=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if not version_spec:
        version_spec = version

    if depends:
        pkg['depends'] = update_dependencies(pkg['depends'], depends)

    pkg['name'] = f'pytorch3d:{version}'

    pkg['build_args'] = {
        'PYTORCH3D_VERSION': version,
        'PYTORCH3D_VERSION_SPEC': version_spec
    }

    builder = pkg.copy()

    builder['name'] = f'pytorch3d:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'pytorch3d'
        builder['alias'] = 'pytorch3d:builder'

    return pkg, builder

package = [
    pytorch3d('0.8.0', '0.7.9', default=True), # Compatible with CUDA 13 (Spark and Thor)
]
