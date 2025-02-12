
from jetson_containers import L4T_VERSION

def mlc(commit, patch=None, version='0.1', tvm='0.15', llvm=17, requires=None, default=False):
    pkg = package.copy()
  
    if default:
        pkg['alias'] = 'mlc'
        
    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'mlc:{version}'
    pkg['notes'] = f"[mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm/tree/{commit}) commit SHA [`{commit}`](https://github.com/mlc-ai/mlc-llm/tree/{commit})"
    
    pkg['build_args'] = {
        'MLC_VERSION': version,
        'MLC_COMMIT': commit,
        'MLC_PATCH': patch,
        'TVM_VERSION': tvm,
        'LLVM_VERSION': llvm
    }
    
    builder = pkg.copy()
    
    builder['name'] = f'mlc:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    return pkg, builder

package = [
    mlc('51fb0f4', 'patches/51fb0f4.diff', version='0.1.0', tvm='0.15.0', default=(L4T_VERSION.major == 35), requires='==35.*'), # 12/15/2023
    mlc('607dc5a', 'patches/607dc5a.diff', version='0.1.0', tvm='0.15.0', default=(L4T_VERSION.major >= 36), requires='>=36'),  # 02/27/2024
    mlc('3403a4e', 'patches/3403a4e.diff', version='0.1.1', tvm='0.16.0', requires='>=36'),  # 4/15/2024
    mlc('9336b4a', 'patches/9336b4a.diff', version='0.1.2', tvm='0.18.0', requires='>=36'),  # 9/25/2024
    mlc('6da6aca', 'patches/6da6aca.diff', version='0.1.3', tvm='0.18.1', requires='>=36'),  # 10/18/2024
    mlc('385cef2', 'patches/385cef2.diff', version='0.1.4', tvm='0.19.0', requires='>=36'),  # 12/14/2024
    mlc('cf7ae82', 'patches/cf7ae82.diff', version='0.19.0', tvm='0.19.0', requires='>=36', default=True), # 01/09/2025
    mlc('b636b2a', 'patches/b636b2a.diff', version='0.20.0', tvm='0.20.0', requires='>=36'), # 12/06/2025
]

