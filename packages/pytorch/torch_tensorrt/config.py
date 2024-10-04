
from jetson_containers import PYTHON_VERSION, JETPACK_VERSION

if JETPACK_VERSION.major >= 5:
    TORCH_TRT_VERSION = 'v1.4.0'  # build setup has changed > 1.4.0 (still ironing it out on aarch64)
else:
    TORCH_TRT_VERSION = 'v2.4.0'  # compatability with PyTorch 2.4, CUDA 12.4, TensorRT 10.1, Python 3.12
    
package['build_args'] = {
    'PYTHON_VERSION': PYTHON_VERSION,
    'JETPACK_MAJOR': JETPACK_VERSION.major,
    'JETPACK_MINOR': 0 if JETPACK_VERSION.major >= 5 else 6,   # only 5.0 and 4.6 are recognized
    'TORCH_TRT_VERSION': TORCH_TRT_VERSION,  
}
