
def AutoGPTQ(version, branch=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'auto_gptq:{version}'

    if default:
        pkg['alias'] = 'auto_gptq'
    
    if not branch:
        branch = version
        
    pkg['build_args'] = {
        'AUTOGPTQ_VERSION': version,
        'AUTOGPTQ_BRANCH': branch,
    }
    
    return pkg

package = [
    AutoGPTQ('0.7.1', default=False),
    AutoGPTQ('0.8.0', default=True),
]
