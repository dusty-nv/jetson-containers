


def flash_infer(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'flashinfer:{version}'
    
    pkg['build_args'] = {
        'FLASHINFER_VERSION': version,
        'FLASHINFER_VERSION_SPEC': version_spec if version_spec else version,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'flashinfer:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'flashinfer'
        builder['alias'] = 'flashinfer:builder'
        
    return pkg, builder

package = [
    flash_infer('0.2.3', '0.2.2.post1', default=False),
    flash_infer('0.2.4', '0.2.4', default=True),
]

