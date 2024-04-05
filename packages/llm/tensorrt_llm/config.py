from jetson_containers import CUDA_ARCHITECTURES, CUDA_VERSION, update_dependencies
from packaging.version import Version
   
def tensorrt_llm(version, branch=None, patch=None, depends=None, requires=None, default=False):
    trt_llm = package.copy()

    trt_llm['name'] = f'tensorrt_llm:{version}'
    
    if not branch:
        branch = 'v' + version
        if len(branch.split('.')) < 3:
            branch = branch + '.0'

    if not patch:
        patch = 'patches/empty.diff'
        
    trt_llm['build_args'] = {
        'TRT_LLM_VERSION': version,
        'TRT_LLM_BRANCH': branch,
        'TRT_LLM_PATCH': patch,
        'CUDA_ARCHS': ';'.join([f'{x}-real' for x in CUDA_ARCHITECTURES]),
    }

    if depends:
        trt_llm['depends'] = update_dependencies(trt_llm['depends'], depends)
        
    #if tensorrt:
    #    trt_llm['depends'] = [f"tensorrt:{tensorrt}" if x=='tensorrt' else x for x in trt_llm['depends']]
        
    if requires:
        trt_llm['requires'] = requires
        
    builder = trt_llm.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    if default:
        trt_llm['alias'] = 'tensorrt_llm'
        builder['alias'] = 'tensorrt_llm:builder'
        
    return trt_llm, builder

package = [
    tensorrt_llm('0.9.dev', '118b3d7', patch='patches/118b3d7.diff', requires=['==r36.*', '>=cu124'], default=True),
    tensorrt_llm('0.5', patch='patches/0.5.diff', requires=['==r36.*', '==cu122'], default=True),
]
