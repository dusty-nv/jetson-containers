import os
import copy

from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES, find_container, github_latest_commit, log_debug

repo = 'mlc-ai/mlc-llm'

package['build_args'] = {
    'MLC_REPO': repo,
    'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
}

def mlc(version, patch, llvm=17, tag=None, requires=None, default=False):
    build = package.copy()
    deploy = package.copy()
    
    if default:
        build['alias'] = 'mlc:builder'
        deploy['alias'] = 'mlc'
        
    if requires:
        build['requires'] = requires
        deploy['requires'] = requires   
        
    if not tag:
        tag = version

    build['name'] = f'mlc:{tag}-builder'
    deploy['name'] = f'mlc:{tag}'
    
    build['dockerfile'] = 'Dockerfile.builder'
    
    build['build_args'] = {
        'MLC_REPO': repo,
        'MLC_VERSION': version,
        'MLC_PATCH': patch,
        'LLVM_VERSION': llvm,
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
    }
    
    deploy['build_args'] = {
        'BUILD_IMAGE': find_container(build['name'])
    }
    
    build['notes'] = f"[{repo}](https://github.com/{repo}/tree/{version}) commit SHA [`{version}`](https://github.com/{repo}/tree/{version})"
    deploy['notes'] = build['notes']
    
    return build, deploy

latest_sha = github_latest_commit(repo, branch='main')
log_debug('-- MLC latest commit:', latest_sha)

#default_dev=(L4T_VERSION.major >= 36)

package = [
    mlc(latest_sha, 'patches/51fb0f4.diff', llvm=18, tag='dev'), #, default=default_dev),
    mlc('9bf5723', 'patches/9bf5723.diff', llvm=17, requires='==35.*'), # 10/20/2023
    mlc('51fb0f4', 'patches/51fb0f4.diff', llvm=17, default=True), # 12/15/2023
]