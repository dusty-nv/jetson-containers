
def torchao(version, requires=None, default=False):
    pkg = package.copy()
    
    pkg['name'] = f"torchao:{version.split('-')[0]}"  # remove any -rc* suffix

    if requires:
        pkg['requires'] = requires
   
    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {
        'TORCHAO_VERSION': version,
    }
    
    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if default:
        pkg['alias'] = 'torchao'
        builder['alias'] = 'torchao:builder'

    return pkg, builder
    
 
package = [
    torchao('0.7.0', requires='==36.*'),
    torchao('0.8.0', requires='==36.*'),
    torchao('0.11.0', requires='==36.*', default=True),
]
