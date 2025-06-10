def fourk4D(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'4k4D:{version}'

    pkg['build_args'] = {
        '4k4D_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'4k4D:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = '4k4D'
        builder['alias'] = '4k4D:builder'

    return pkg, builder

package = [
    fourk4D('0.0.0', default=True)
]
