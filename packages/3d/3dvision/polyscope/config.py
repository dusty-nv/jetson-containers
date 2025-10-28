def polyscope(version, version_spec, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'polyscope:{version}'

    pkg['build_args'] = {
        'POLYSCOPE_VERSION': version,
        'POLYSCOPE_VERSION_SPEC': version_spec if version_spec else version,

    }

    builder = pkg.copy()

    builder['name'] = f'polyscope:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'polyscope'
        builder['alias'] = 'polyscope:builder'

    return pkg, builder

package = [
    polyscope('2.6.0', version_spec='2.5.0', default=True),
]
