def fourk4D(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'4k4d:{version}'

    pkg['build_args'] = {
        'FOURkFOUR_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'4k4d:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = '4k4d'
        builder['alias'] = '4k4d:builder'

    return pkg, builder

package = [
    fourk4D('0.0.0', default=True)
]
