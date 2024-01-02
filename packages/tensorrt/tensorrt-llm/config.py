
from jetson_containers import CUDA_ARCHITECTURES

import copy

def tensorrt_llm(version, tensorrt=None, patch=None, requires=None, default=False):
    trt_llm = copy.deepcopy(package)

    trt_llm['name'] = f'tensorrt-llm:{version}'
    
    if len(version.split('.')) < 3:
        version = version + '.0'
        
    if not patch:
        patch = 'patches/empty.diff'
        
    trt_llm['build_args'] = {
        'TRT_LLM_VERSION': version,
        'TRT_LLM_PATCH': patch,
        'CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    }

    if patch:
        trt_llm['build_args']['TRT_LLM_PATCH'] = patch
        
    if tensorrt:
        trt_llm['depends'][0] = f'tensorrt:{tensorrt}'
        
    if requires:
        trt_llm['requires'] = requires
        
    if default:
        trt_llm['alias'] = 'tensorrt-llm'
        
    return trt_llm
  
package = [
    tensorrt_llm('0.5', tensorrt='8.6', patch='patches/0.5.diff', requires='==36.*', default=True),
]
