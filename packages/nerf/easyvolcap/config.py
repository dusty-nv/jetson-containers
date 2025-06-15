
def easyvolcap(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'easyvolcap:{version}'

    pkg['build_args'] = {
        'EASYVOLCAP_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'easyvolcap:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'easyvolcap'
        builder['alias'] = 'easyvolcap:builder'

    return pkg, builder

package = [
    easyvolcap('0.0.0', default=True)
]
