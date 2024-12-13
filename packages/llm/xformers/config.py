
def xformers(version, requires=None, default=True):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'xformers:{version}'
    
    pkg['build_args'] = {
        'XFORMERS_VERSION': version,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'xformers:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'xformers'
        builder['alias'] = 'xformers:builder'
        
    return pkg, builder

package = [
    xformers('0.0.26', requires='<=cu122'),
    xformers('0.0.28', requires='>=cu124'),
]
