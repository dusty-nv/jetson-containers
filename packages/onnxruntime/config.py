
from jetson_containers import L4T_VERSION

# https://elinux.org/Jetson_Zoo#ONNX_Runtime
if L4T_VERSION.major >= 36:
    ONNXRUNTIME_URL="https://nvidia.box.com/shared/static/i7n40ki3pl2x57vyn4u7e9asyiqlnl7n.whl"
    ONNXRUNTIME_WHL="onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl"
elif L4T_VERSION.major >= 34:
    ONNXRUNTIME_URL="https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6kpcuwzd.whl"
    ONNXRUNTIME_WHL="onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl"
else:
    # onnxruntime >= 1.16 drops support for gcc7/Python 3.6 (JetPack 4)
    # onnxruntime >= 1.14 too few arguments to function cudaStreamWaitEvent
    # onnxruntime <= 1.13 doesn't need or support --allow_running_as_root
    # onnxruntime >= 1.12 error: 'getBuilderPluginRegistry' is not a member of 'nvinfer1'
    # onnxruntime >= 1.11 error: NvInferSafeRuntime.h: No such file or directory (missing from tensorrt.csv)
    #if L4T_VERSION.major <= 32:
    #    package['build_args']['ONNXRUNTIME_VERSION'] = 'v1.10.0'
    #    package['build_args']['ONNXRUNTIME_FLAGS'] = ''
    ONNXRUNTIME_URL="https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl"
    ONNXRUNTIME_WHL="onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl"
    
package['build_args'] = {
    'ONNXRUNTIME_URL': ONNXRUNTIME_URL,
    'ONNXRUNTIME_WHL': ONNXRUNTIME_WHL,
}
