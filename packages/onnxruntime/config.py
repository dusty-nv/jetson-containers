
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

package['build_args'] = {
    'CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
}

# onnxruntime >= 1.16 drops support for gcc7/Python 3.6 (JetPack 4)
# onnxruntime >= 1.14 too few arguments to function cudaStreamWaitEvent
# onnxruntime <= 1.13 doesn't need or support --allow_running_as_root
# onnxruntime >= 1.12 error: 'getBuilderPluginRegistry' is not a member of 'nvinfer1'
# onnxruntime >= 1.11 error: NvInferSafeRuntime.h: No such file or directory (missing from tensorrt.csv)
if L4T_VERSION.major <= 32:
    package['build_args']['ONNXRUNTIME_VERSION'] = 'v1.10.0'
    package['build_args']['ONNXRUNTIME_FLAGS'] = ''
