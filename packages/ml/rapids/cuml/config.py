
from jetson_containers import CUDA_ARCHITECTURES

package['build_args'] = {
    'CUML_VERSION': 'v26.04.00',  # newer versions require CUDA >= 11.4 (this is a version with some patches in dustynv/cudf fork)
    'CUML_CMAKE_CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
}
