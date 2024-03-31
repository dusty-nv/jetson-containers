
from jetson_containers import L4T_VERSION

def mlc(commit, patch=None, version='0.1', tvm='0.15', llvm=17, tag=None, requires=None, default=False):
    pkg = package.copy()
  
    if default:
        pkg['alias'] = 'mlc'
        
    if requires:
        pkg['requires'] = requires   
        
    if not tag:
        tag = commit

    pkg['name'] = f'mlc:{tag}'
    pkg['notes'] = f"[mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm/tree/{commit}) commit SHA [`{commit}`](https://github.com/mlc-ai/mlc-llm/tree/{commit})"
    
    pkg['build_args'] = {
        'MLC_VERSION': version,
        'MLC_COMMIT': commit,
        'MLC_PATCH': patch,
        'TVM_VERSION': tvm,
        'LLVM_VERSION': llvm
    }

    return pkg

package = [
    mlc('51fb0f4', 'patches/51fb0f4.diff', tvm='0.12', default=(L4T_VERSION.major == 35)), # 12/15/2023
    mlc('607dc5a', 'patches/607dc5a.diff', tvm='0.15', default=(L4T_VERSION.major >= 36), requires='>=36'),  # 02/27/2024
]

#latest_sha = github_latest_commit(repo, branch='main')
#log_debug('-- MLC latest commit:', latest_sha)

'''
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
'''