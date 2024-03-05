import os
import copy

from jetson_containers import L4T_VERSION, PYTHON_VERSION, CUDA_ARCHITECTURES, find_container, github_latest_commit, log_debug

repo = 'mlc-ai/mlc-llm'

package['build_args'] = {
    'MLC_REPO': repo,
    'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
}

def mlc(version, patch, llvm=17, tag=None, requires=None, default=False):
    builder = package.copy()
    runtime = package.copy()
    
    if default:
        builder['alias'] = 'mlc:builder'
        runtime['alias'] = 'mlc'
        
    if requires:
        builder['requires'] = requires
        runtime['requires'] = requires   
        
    if not tag:
        tag = version

    builder['name'] = f'mlc:{tag}-builder'
    runtime['name'] = f'mlc:{tag}'
    
    builder['dockerfile'] = 'Dockerfile.builder'
    
    builder['build_args'] = {
        'MLC_REPO': repo,
        'MLC_VERSION': version,
        'MLC_PATCH': patch,
        'LLVM_VERSION': llvm,
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
    }
    
    runtime['build_args'] = {
        'BUILD_IMAGE': find_container(builder['name']),
        'PYTHON_VERSION': str(PYTHON_VERSION),
        'MLC_REPO': repo,
        'MLC_VERSION': version,
        'MLC_PATCH': patch,
    }
    
    builder['notes'] = f"[{repo}](https://github.com/{repo}/tree/{version}) commit SHA [`{version}`](https://github.com/{repo}/tree/{version})"
    runtime['notes'] = builder['notes']
    
    return builder, runtime

#latest_sha = github_latest_commit(repo, branch='main')
#log_debug('-- MLC latest commit:', latest_sha)

package = [
    mlc('731616e', 'patches/3feed05.diff', tag='dev'),
    mlc('9bf5723', 'patches/9bf5723.diff', requires='==35.*'), # 10/20/2023
    mlc('51fb0f4', 'patches/51fb0f4.diff', default=(L4T_VERSION.major == 35)), # 12/15/2023
    mlc('3feed05', 'patches/3feed05.diff', requires='>=36'), # 02/08/2024
    #mlc('6cf63bb', 'patches/3feed05.diff', requires='>=36'),  # 02/16/2024
    #mlc('c30348a', 'patches/3feed05.diff', requires='>=36'),  # 02/19/2024
    #mlc('a2d9eea', 'patches/3feed05.diff', requires='>=36'),  # 02/19/2024
    mlc('5584cac', 'patches/3feed05.diff', requires='>=36'),   # 02/21/2024
    mlc('607dc5a', 'patches/607dc5a.diff', default=(L4T_VERSION.major >= 36), requires='>=36'),  # 02/27/2024
    mlc('1f70d71', 'patches/1f70d71.diff', requires='>=36'),   # 02/29/2024
    mlc('731616e', 'patches/1f70d71.diff', requires='>=36'),   # 03/03/2024
]