
def diffusers(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'diffusers:{version}'

    pkg['build_args'] = {
        'DIFFUSERS_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'diffusers:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'diffusers'
        builder['alias'] = 'diffusers:builder'

    return pkg, builder

package = [
    diffusers('0.36.0', default=True),
]
