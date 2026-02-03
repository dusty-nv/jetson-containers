
def ktransformers(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'ktransformers:{version}'

    pkg['build_args'] = {
        'KTRANSFORMERS_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'ktransformers:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'ktransformers'
        builder['alias'] = 'ktransformers:builder'

    return pkg, builder

package = [
    ktransformers(version='0.5.1', default=True),
]
