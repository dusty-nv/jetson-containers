from jetson_containers import CUDA_ARCHITECTURES, CUDA_VERSION, update_dependencies
from packaging.version import Version

def tensorrt_llm(version, branch=None, patch=None, src=None, depends=None, requires=None, default=False):
    trt_llm = package.copy()

    trt_llm['name'] = f'tensorrt_llm:{version}'

    if not branch:
        branch = 'v' + version
        if len(branch.split('.')) < 3:
            branch = branch + '.0'

    if not patch:
        patch = 'patches/empty.diff'

    if not src:
        src = 'sources/empty.tar.gz'

    trt_llm['build_args'] = {
        'TRT_LLM_VERSION': version,
        'TRT_LLM_BRANCH': branch,
        'TRT_LLM_SOURCE': src,
        'TRT_LLM_PATCH': patch,
        'CUDA_ARCHS': ';'.join([f'{x}-real' for x in CUDA_ARCHITECTURES]),
        'CUDA_VERSION': CUDA_VERSION,
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
    tensorrt_llm('0.12', branch='v0.12.0-jetson', patch='patches/pybind11_python_fix.diff', depends=['cutlass','torch2trt', 'pybind11'], requires='<cu128', default=False),
    tensorrt_llm('1.1.0', depends=['xgrammar:0.1.25'], requires='>=cu126', default=True)]
