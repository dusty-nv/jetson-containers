
def exllama(version, branch=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'exllama:{version}'

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
    #exllama('0.0.16', default=True),
    exllama('0.0.15', default=True),
]
