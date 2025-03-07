
def sglang(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    if not version_spec:
        version_spec = version

    pkg['name'] = f'sglang:{version}'

    pkg['build_args'] = {
        'SGLANG_VERSION': version,
        'SGLANG_VERSION_SPEC': version_spec,
    }

    builder = pkg.copy()

    builder['name'] = f'sglang:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'sglang'
        builder['alias'] = 'sglang:builder'

    return pkg, builder

package = [
    sglang('0.4.4', '0.4.3.post2', default=True),
]
