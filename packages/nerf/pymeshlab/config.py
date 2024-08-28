def pymeshlab(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'pymeshlab:{version}'

    pkg['build_args'] = {
        'PYMESHLAB_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'pymeshlab:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'pymeshlab'
        builder['alias'] = 'pymeshlab:builder'

    return pkg, builder

package = [
    pymeshlab('2023.12', default=True),
]