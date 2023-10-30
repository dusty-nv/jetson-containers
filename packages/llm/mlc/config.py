import os
import copy

from jetson_containers import CUDA_ARCHITECTURES, github_latest_commit

repo = 'mlc-ai/mlc-llm'

package['build_args'] = {
    'MLC_REPO': repo,
    'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
}

def mlc(version, patch, tag=None, default=False):
    pkg = copy.deepcopy(package)
    
    if default:
        pkg['alias'] = 'mlc'
        
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
print('-- MLC latest commit:', latest_sha)

package = [
    mlc(latest_sha, 'patches/9166edb.diff', tag='dev'), # patched as of 10/29/2023
    mlc('9bf5723', 'patches/9bf5723.diff', default=True), # 10/20/2023
]