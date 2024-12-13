def polyscope(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'polyscope:{version}'

    pkg['build_args'] = {
        'POLYSCOPE_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'polyscope:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'polyscope'
        builder['alias'] = 'polyscope:builder'

    return pkg, builder

package = [
    polyscope('2.3.0'),
    polyscope('2.4.0', default=True),
]
