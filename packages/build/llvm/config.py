

def llvm(version):
    """
    Container with updated LLVM and clang toolchains installed.
    The default is LLVM 19 and is linked to /usr/bin/llvm-config
    """
    pkg = package.copy()

    pkg['name'] = f'llvm:{version}'
    pkg['build_args'] = {'LLVM_VERSION': version}
    
    if version == 19: # current stable
        pkg['alias'] = 'llvm'

    return pkg

package = [
    llvm(i) for i in range(10,20) # llvm 10 was Ubuntu 20.04 / JP5
]
