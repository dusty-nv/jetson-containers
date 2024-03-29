
def torchvision(version, pytorch=None, requires=None, default=False):
    pkg = package.copy()
    
    pkg['name'] = f"torchvision:{version}"
    
    if pytorch:
        pkg['depends'] = [f"pytorch:{pytorch}" if x=='pytorch' else x for x in pkg['depends']]
        
    if requires:
        pkg['requires'] = requires
        
    if default:
        pkg['alias'] = 'torchvision'
     
    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {
        'TORCHVISION_VERSION': version,
    }

    return pkg
    
 
package = [
    # JetPack 6
    torchvision('0.16.2', pytorch='2.1', requires='==36.*', default=True),
    torchvision('0.17.2', pytorch='2.2', requires='==36.*', default=False),
    
    # JetPack 5
    torchvision('0.15.1', requires='==35.*', default=True),
    
    # JetPack 4
    torchvision('0.11.1', requires='==32.*', default=True),
]
