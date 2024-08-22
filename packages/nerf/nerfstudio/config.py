def nerfstudio(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'nerfstudio:{version}'

    pkg['build_args'] = {
        'NERFSTUDIO_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'nerfstudio:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'nerfstudio'
        builder['alias'] = 'nerfstudio:builder'

    return pkg, builder

package = [
    nerfstudio('0.3.2', default=True),
]