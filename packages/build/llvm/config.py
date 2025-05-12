
LLVM_STABLE=19
LLVM_LATEST=20

def llvm(version):
    """
    Container with updated LLVM and clang toolchains installed.
    The default is LLVM 19 and is linked to /usr/bin/llvm-config
    """
    pkg = package.copy()

    pkg['name'] = f'llvm:{version}'
    pkg['alias'] = [f'clang:{version}']

    pkg['build_args'] = {'LLVM_VERSION': version}
    
    if version == LLVM_STABLE:
        pkg['alias'] += ['llvm', 'clang']

    return pkg

package = [
    llvm(i) for i in range(10,LLVM_LATEST+1) # llvm 10 was Ubuntu 20.04 / JP5
]
