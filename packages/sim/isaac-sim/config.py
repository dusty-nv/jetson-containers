
def isaacsim(version, default=False):
    pkg = package.copy()
    pkg['name'] = f'isaacsim:{version}'
        
    pkg['build_args'] = {
        'ISAACSIM_VERSION': version,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'isaacsim:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'isaacsim'
        builder['alias'] = 'isaacsim:builder'

    return pkg, builder

package = [
    isaacsim('5.0.0', default=True),
]
