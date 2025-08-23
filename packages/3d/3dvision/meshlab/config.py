def meshlab(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'meshlab:{version}'

    pkg['build_args'] = {
        'MESHLAB_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'meshlab:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'meshlab'
        builder['alias'] = 'meshlab:builder'

    return pkg, builder

package = [
    meshlab('MeshLab-2023.12', default=False),
    meshlab('MeshLab-2025.07', default=True),
]
