
def awq(version, kernels=None, branch=None, default=False):
    pkg = package.copy()
    pkg['name'] = f'awq:{version}'

    if default:
        pkg['alias'] = 'awq'
    
    if not branch:
        branch = version
        
    pkg['build_args'] = {
        'AWQ_BRANCH': branch,
        'AWQ_VERSION': version,
        'AWQ_KERNEL_VERSION': kernels,
    }
    
    return pkg

package = [
    awq('0.1.0', kernels='0.0.0', branch='main', default=True),
]
