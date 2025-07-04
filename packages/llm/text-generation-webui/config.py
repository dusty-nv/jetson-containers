
from jetson_containers import L4T_VERSION

def oobabooga(version, branch=None, tag=None, sha=None, build_args=None, default=False):
    twu = package.copy()
    
    if default:
        twu['alias'] = twu['name']
        
    twu['name'] = f"{twu['name']}:{version}"
    
    if branch:
        ref = f"refs/heads/{branch}"
    elif tag:
        ref = f"refs/tags/{tag}"
    elif sha:
        ref = f"commits/{sha}"
    else:
        raise ValueError("either branch, tag, or sha needs to be specified")
        
    twu['build_args'] = {
        'OOBABOOGA_REF': ref,
        'OOBABOOGA_SHA': ref.split('/')[-1]
    }
    
    if build_args:
        twu['build_args'].update(build_args)
        
    #if L4T_VERSION.major <= 35:
    #    twu['depends'] = twu['depends'] + ['bitsandbytes']
    
    # auto_awq is depracated
    # if L4T_VERSION.major >= 36:
    #     twu['depends'] = twu['depends'] + ['auto_awq']
        
    return twu
    
package = [
    oobabooga('main', branch='main', default=True),
    oobabooga('1.7', tag='v1.7', build_args=dict(LD_PRELOAD_LIBS='/usr/local/lib/python3.8/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0')),
    oobabooga('6a7cd01', sha='6a7cd01ebf8021a8ee6da094643f09da41516ccd'), # last commit to support original server API
]