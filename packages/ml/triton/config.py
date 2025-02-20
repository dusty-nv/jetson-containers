def triton(version, branch=None, requires=None, default=False):
    pkg = package.copy()

    if not branch:
        branch = f'v{version}'
        
    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'triton:{version}'
    
    pkg['build_args'] = {
        'TRITON_VERSION': version,
        'TRITON_BRANCH': branch,
    }
    
    builder = pkg.copy()
    builder['name'] += '-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}
    
    if default:
        pkg['alias'] = 'triton'
        builder['alias'] = 'triton:builder'
        
    return pkg, builder

package = [
    triton('3.2.0', branch='release/3.3.x'),
    triton('3.2.0', branch='release/3.2.x', default=True),
    triton('3.1.0', branch='release/3.1.x'),
    triton('3.0.0', branch='release/3.0.x'),
]

