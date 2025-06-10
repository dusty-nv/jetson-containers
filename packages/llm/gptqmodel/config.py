
def gptqmodel(version, branch=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'gptqmodel:{version}'

    if not branch:
        branch = version
        
    pkg['build_args'] = {
        ''
        'GPTQMODEL_VERSION': version,
        'GPTQMODEL_BRANCH': branch,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'auto_gptq:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'auto_gptq'
        builder['alias'] = 'auto_gptq:builder'
        
    return pkg, builder


package = [
    gptqmodel('3.0.1', default=True),
]
