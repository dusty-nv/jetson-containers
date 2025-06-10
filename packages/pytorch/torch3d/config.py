
def pytorch3d(version, requires=None, default=False):
    pkg = package.copy()
    
    pkg['name'] = f"pytorch3d:{version.split('-')[0]}"  # remove any -rc* suffix

    if requires:
        pkg['requires'] = requires
   
    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {
        'PYTORCH3D_VERSION': version,
    }
    
    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if default:
        pkg['alias'] = 'pytorch3d'
        builder['alias'] = 'pytorch3d:builder'

    return pkg, builder
    
 
package = [
    pytorch3d('0.7.8', requires='==36.*', default=False),
    pytorch3d('0.7.9', requires='==36.*', default=True),
]
