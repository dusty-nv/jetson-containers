
def torchaudio(version, pytorch=None, requires=None, default=False):
    pkg = package.copy()
    
    pkg['name'] = f"torchaudio:{version}"
    
    if pytorch:
        pkg['depends'] = [f"pytorch:{pytorch}" if x=='pytorch' else x for x in pkg['depends']]
        
    if requires:
        pkg['requires'] = requires
        
    if default:
        pkg['alias'] = 'torchaudio'
     
    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {
        'TORCHAUDIO_VERSION': version,
    }

    return pkg

package = [
    # JetPack 6
    torchaudio('2.1.0', pytorch='2.1', requires='==36.*', default=True),
    torchaudio('2.2.2', pytorch='2.2', requires='==36.*', default=False),
    
    # JetPack 5
    torchaudio('2.0.1', requires='==35.*', default=True),
    
    # JetPack 4
    torchaudio('0.10.0', requires='==32.*', default=True),
]
