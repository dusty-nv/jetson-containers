import os
import copy

from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES, github_latest_commit, log_debug

repo = 'mlc-ai/mlc-llm'

package['build_args'] = {
    'MLC_REPO': repo,
    'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
}

def mlc(version, patch, tag=None, requires=None, default=False):
    pkg = copy.deepcopy(package)
    
    if default:
        pkg['alias'] = 'mlc'
        
    if requires:
        pkg['requires'] = requires
        
    if not tag:
        tag = version
        
    pkg['name'] = f'mlc:{tag}'

    pkg['build_args'].update({
        'MLC_VERSION': version,
        'MLC_PATCH': patch,
    })
    
    pkg['notes'] = f"[{repo}](https://github.com/{repo}/tree/{version}) commit SHA [`{version}`](https://github.com/{repo}/tree/{version})"
    
    return pkg

latest_sha = github_latest_commit(repo, branch='main')
log_debug('-- MLC latest commit:', latest_sha)

default_dev=(L4T_VERSION.major >= 36)

package = [
    mlc(latest_sha, 'patches/empty.diff', tag='dev', default=default_dev),
    mlc('9bf5723', 'patches/9bf5723.diff', requires='==35.*', default=not default_dev), # 10/20/2023
]