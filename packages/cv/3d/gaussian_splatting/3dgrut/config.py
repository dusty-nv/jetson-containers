
def threedgrut(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'3dgrut:{version}'

    pkg['build_args'] = {
        'THREEGRUT_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'3dgrut:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = '3dgrut'
        builder['alias'] = '3dgrut:builder'

    return pkg, builder

package = [
    threedgrut('2.0.0', default=True)
]
