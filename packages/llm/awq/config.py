
def awq(version, kernels=None, branch=None, default=False):
    pkg = package.copy()
    pkg['name'] = f'awq:{version}'

    if not branch:
        branch = version
        
    pkg['build_args'] = {
        'AWQ_REPO': 'mit-han-lab/llm-awq',
        'AWQ_BRANCH': branch,
        'AWQ_VERSION': version,
        'AWQ_KERNEL_VERSION': kernels,
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'awq:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'awq'
        builder['alias'] = 'awq:builder'

    return pkg, builder

package = [
    awq('0.1.0', kernels='0.0.0', branch='main', default=True),
]
