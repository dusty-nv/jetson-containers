
from jetson_containers import L4T_VERSION, ARCH, CUDA_ARCH_LIST

OPENCV_VERSION = "4.5.0"

if ARCH == 'aarch64':
    CUDA_ARCH_BIN = ','.join(CUDA_ARCH_LIST)
    ENABLE_NEON = "ON"
elif ARCH == 'x86_64':
    CUDA_ARCH_BIN = ''
    ENABLE_NEON = "OFF"
    
package['build_args'] = {
    'OPENCV_VERSION': '4.5.0',
    'CUDA_ARCH_BIN': CUDA_ARCH_BIN,
    'ENABLE_NEON': ENABLE_NEON
}

package['depends'] = ['cmake', 'python']
