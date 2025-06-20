
def nvdiffrast(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'nvdiffrast:{version}'

    pkg['build_args'] = {
        'NVDIFFRAST_VERSION': version,

    }

    builder = pkg.copy()

    builder['name'] = f'nvdiffrast:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'nvdiffrast'
        builder['alias'] = 'nvdiffrast:builder'

    return pkg, builder

package = [
    nvdiffrast('0.3.4', default=True),
]
