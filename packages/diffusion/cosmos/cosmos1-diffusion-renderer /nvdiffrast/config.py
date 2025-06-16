
def nvdiffrast(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'nvdiffrast:{version}'

    pkg['build_args'] = {
        'nvdiffrast_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'nvdiffrast:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'nvdiffrast'
        builder['alias'] = 'nvdiffrast:builder'

    return pkg, builder

package = [
    nvdiffrast('0.30.2'),
    nvdiffrast('0.31.0'),
    nvdiffrast('0.32.3'),
    nvdiffrast('0.33.1'),
    nvdiffrast('0.34.0', default=True),
]
