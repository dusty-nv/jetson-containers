
from jetson_containers import L4T_VERSION, SYSTEM_ARCH, CUDA_ARCHITECTURES


def build_opencv(version, requires=None):
    cv = package.copy()
    
    if SYSTEM_ARCH == 'aarch64':
        CUDA_ARCH_BIN = ','.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
        ENABLE_NEON = "ON"
    elif SYSTEM_ARCH == 'x86_64':
        CUDA_ARCH_BIN = ''
        ENABLE_NEON = "OFF"

    if L4T_VERSION.major >= 36:
        ENABLE_OPENGL = "OFF"
    else:
        ENABLE_OPENGL = "ON"
        
    cv['build_args'] = {
        'OPENCV_VERSION': version,
        'CUDA_ARCH_BIN': CUDA_ARCH_BIN,
        'ENABLE_NEON': ENABLE_NEON,
        'ENABLE_OPENGL': ENABLE_OPENGL,
    }

    cv['name'] = f'opencv:{version}-builder'
    cv['group'] = 'core'
    cv['depends'] = ['cuda', 'cudnn', 'cmake', 'python']
    cv['notes'] = 'the built packages are bundled into a .tar.gz under /opt'

    if requires:
        cv['requires'] = requires
        
    return cv
    
    
package = [
    build_opencv('4.8.1', requires='==36.*'),
    build_opencv('4.5.0', requires='<=35')
]
