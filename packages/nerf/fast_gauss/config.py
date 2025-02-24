
def fast_gauss(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'fast_gauss:{version}'

    pkg['build_args'] = {
        'FAST_GAUSS_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'fast_gauss:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'fast_gauss'
        builder['alias'] = 'fast_gauss:builder'

    return pkg, builder

package = [
    fast_gauss('1.0.0', default=True)
]
