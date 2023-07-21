
from jetson_containers import CUDA_ARCHITECTURES

package['build_args'] = {
    'CUDF_VERSION': 'v21.10.02',  # newer versions require CUDA >= 11.5 (this is a version with some patches in dustynv/cudf fork)
    'CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
}
