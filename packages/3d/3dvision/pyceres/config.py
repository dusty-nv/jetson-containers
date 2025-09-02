
def pyceres(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'pyceres:{version}'

    pkg['build_args'] = {
        'PYCERES_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'pyceres:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'pyceres'
        builder['alias'] = 'pyceres:builder'

    return pkg, builder

package = [
    pyceres('2.6', default=True)
]
