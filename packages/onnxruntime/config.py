
from jetson_containers import L4T_VERSION, CUDA_ARCH_LIST_INT

package['build_args'] = {
    'CUDA_ARCHITECTURES': ';'.join([str(c) for c in CUDA_ARCH_LIST_INT]),
}

# onnxruntime >= 1.16 drops support for gcc7/Python 3.6 (JetPack 4)
# onnxruntime >= 1.14 too few arguments to function cudaStreamWaitEvent
# onnxruntime <= 1.13 doesn't need/support --allow_running_as_root
if L4T_VERSION.major <= 32:
    package['build_args']['ONNXRUNTIME_VERSION'] = 'v1.13.1'
    package['build_args']['ONNXRUNTIME_FLAGS'] = ''
