
def minference(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if not version_spec:
        version_spec = version

    pkg['name'] = f'minference:{version}'

    pkg['build_args'] = {
        'MINFERENCE_VERSION': version,
        'MINFERENCE_VERSION_SPEC': version_spec,
    }

    builder = pkg.copy()

    builder['name'] = f'minference:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'minference'
        builder['alias'] = 'minference:builder'

    return pkg, builder

package = [
    minference('0.1.7', '0.1.7', default=True),
]
