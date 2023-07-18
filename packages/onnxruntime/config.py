
from jetson_containers import L4T_VERSION, CUDA_ARCH_LIST_INT

ONNXRUNTIME_VERSION = 'main'
CUDA_ARCHITECTURES = ';'.join([str(c) for c in CUDA_ARCH_LIST_INT])

# onnxruntime >= 1.16 drops support for gcc7/Python 3.6 (JetPack 4)
if L4T_VERSION.major <= 32:
    ONNXRUNTIME_VERSION = 'v1.15.1'
    
package['build_args'] = {
    'ONNXRUNTIME_VERSION': ONNXRUNTIME_VERSION,
    'CUDA_ARCHITECTURES': CUDA_ARCHITECTURES,
}
