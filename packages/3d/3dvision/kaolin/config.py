
def kaolin(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'kaolin:{version}'

    pkg['build_args'] = {
        'KAOLIN_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'kaolin:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'kaolin'
        builder['alias'] = 'kaolin:builder'

    return pkg, builder

package = [
    kaolin('0.19.0', default=True)
]
