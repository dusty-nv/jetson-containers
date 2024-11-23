
def AutoGPTQ(version, branch=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'auto_gptq:{version}'

    if not branch:
        branch = version
        
    pkg['build_args'] = {
        'AUTOGPTQ_VERSION': version,
        'AUTOGPTQ_BRANCH': branch,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'auto_gptq:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'auto_gptq'
        builder['alias'] = 'auto_gptq:builder'
        
    return pkg, builder


package = [
    AutoGPTQ('0.7.1', default=False),
    AutoGPTQ('0.8.0', default=True),
]
