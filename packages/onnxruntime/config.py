
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

package['build_args'] = {
    'CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
}

# Source: https://elinux.org/Jetson_Zoo#ONNX_Runtime
if L4T_VERSION.major <= 34:
    # onnxruntime 1.11.0
    package['build_args']['ONNXRUNTIME_WHEEL'] = 'https://nvidia.box.com/shared/static/2sv2fv1wseihaw8ym0d4srz41dzljwxh.whl'
    package['build_args']['ONNXRUNTIME_WHEEL_FILENAME'] = 'onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl'
elif L4T_VERSION.major == 35 and L4T_VERSION.minor == 1:
    # onnxruntime 1.12.1
    package['build_args']['ONNXRUNTIME_WHEEL'] = 'https://nvidia.box.com/shared/static/6pt8royppnkhx4us806s1f02n2kbk4po.whl'
    package['build_args']['ONNXRUNTIME_WHEEL_FILENAME'] = 'onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl'
elif L4T_VERSION.major == 35 and L4T_VERSION.minor == 4 and L4T_VERSION.micro == 1:
    # onnxruntime 1.16.0
    package['build_args']['ONNXRUNTIME_WHEEL'] = 'https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl'
    package['build_args']['ONNXRUNTIME_WHEEL_FILENAME'] = 'onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl'
elif L4T_VERSION.major == 36:
    # onnxruntime 1.17.0
    package['build_args']['ONNXRUNTIME_WHEEL'] = 'https://nvidia.box.com/shared/static/i7n40ki3pl2x57vyn4u7e9asyiqlnl7n.whl'
    package['build_args']['ONNXRUNTIME_WHEEL_FILENAME'] = 'onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl'
