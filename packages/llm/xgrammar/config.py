
def xgrammar(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'xgrammar:{version}'

    pkg['build_args'] = {
        'XGRAMMAR_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'xgrammar:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'xgrammar'
        builder['alias'] = 'xgrammar:builder'

    return pkg, builder

package = [
    xgrammar('0.1.9', default=True),
]
