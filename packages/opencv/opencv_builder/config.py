
from jetson_containers import L4T_VERSION, SYSTEM_ARCH, CUDA_ARCHITECTURES

OPENCV_VERSION = "4.5.0"

if SYSTEM_ARCH == 'aarch64':
    CUDA_ARCH_BIN = ','.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
    ENABLE_NEON = "ON"
elif SYSTEM_ARCH == 'x86_64':
    CUDA_ARCH_BIN = ''
    ENABLE_NEON = "OFF"
    
package['build_args'] = {
    'OPENCV_VERSION': '4.5.0',
    'CUDA_ARCH_BIN': CUDA_ARCH_BIN,
    'ENABLE_NEON': ENABLE_NEON
}

package['depends'] = ['cmake', 'python']
