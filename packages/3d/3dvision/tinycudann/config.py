
def tinycudann(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'tinycudann:{version}'

    pkg['build_args'] = {
        'TINYCUDANN_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'tinycudann:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'off'}}

    if default:
        pkg['alias'] = 'tinycudann'
        builder['alias'] = 'tinycudann:builder'

    return pkg, builder

package = [
    tinycudann('2.0', default=True),
]
