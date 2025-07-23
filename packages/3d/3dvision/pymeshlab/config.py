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
    builder['dockerfile'] = 'Dockerfile.builder'
    builder['build_args'] = {**builder['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'pymeshlab'
        builder['alias'] = 'pymeshlab:builder'

    return pkg, builder

package = [
    pymeshlab('2023.12.post2', default=False),
    pymeshlab('2023.12.post3', default=False),
    pymeshlab('2025.7', default=True),
]
