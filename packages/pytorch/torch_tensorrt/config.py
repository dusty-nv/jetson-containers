from jetson_containers import PYTHON_VERSION, JETPACK_VERSION
from packaging.version import Version
if JETPACK_VERSION >= Version('6.2'):
    TORCH_TRT_VERSION = 'v2.11.0' #'v2.11.0'
    JETPACK_MINOR_VER = JETPACK_VERSION.minor
elif JETPACK_VERSION >= Version('6.2'):
    TORCH_TRT_VERSION = 'v2.8.0' #'v2.7.0'
    JETPACK_MINOR_VER = JETPACK_VERSION.minor
elif JETPACK_VERSION < Version('6.2'):
    TORCH_TRT_VERSION = 'lluo/jp6.1' #'v2.4.0'
    JETPACK_MINOR_VER = JETPACK_VERSION.minor
elif JETPACK_VERSION.major >= 5:
    TORCH_TRT_VERSION = 'v1.4.0'  # build setup has changed > 1.4.0 (still ironing it out on aarch64)
    JETPACK_MINOR_VER = 0
else:
    TORCH_TRT_VERSION = 'v1.0.0'  # compatability with PyTorch 2.4, CUDA 12.4, TensorRT 10.1, Python 3.12
    JETPACK_MINOR_VER = 6

package['build_args'] = {
    'PYTHON_VERSION': PYTHON_VERSION,
    'JETPACK_MAJOR': JETPACK_VERSION.major,
    'JETPACK_MINOR': JETPACK_MINOR_VER,   # only 6.0, 6.1, 5.0 and 4.6 are recognized
    'TORCH_TRT_VERSION': TORCH_TRT_VERSION,
}
