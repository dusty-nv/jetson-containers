
def exllama(version, branch=None, requires=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'exllama:{version}'

    if requires:
        pkg['requires'] = requires
        
    if default:
        pkg['alias'] = 'exllama'
    
    if not branch:
        branch = version
        
    pkg['build_args'] = {
        'EXLLAMA_VERSION': version,
        'EXLLAMA_BRANCH': branch,
    }
    
    return pkg

package = [
    #exllama('0.0.16', requires='>=36', default=True),
    exllama('0.0.15', requires='>=36', default=True),
    exllama('0.0.14', requires='==35.*', default=True),
]
