
def nerfview(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'nerfview:{version}'

    pkg['build_args'] = {
        'NERFVIEW_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'nerfview:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}
    builder['depends'] = builder['depends'] + ['gsplat']

    if default:
        pkg['alias'] = 'nerfview'
        builder['alias'] = 'nerfview:builder'

    return pkg, builder

package = [
    nerfview('0.1.5', default=True),
]
