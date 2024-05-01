
from jetson_containers import L4T_VERSION

def NanoLLM(version, branch=None, requires=None, default=False):
    pkg = package.copy()
  
    pkg['name'] = f"nano_llm:{version}"

    if default:
        pkg['alias'] = 'nano_llm'
        
    if requires:
        pkg['requires'] = requires   

    if L4T_VERSION.major >= 36:
        pkg['depends'] = pkg['depends'] + ['xtts']
    
    if not branch:
        branch = version
        
    pkg['build_args'] = {'NANO_LLM_BRANCH': branch}

    return pkg

package = [
    NanoLLM('main', default=True),
    NanoLLM('24.4'),
    NanoLLM('24.4.1'),
]


