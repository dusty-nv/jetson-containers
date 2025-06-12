
from jetson_containers import CUDA_ARCHITECTURES

package['build_args'] = {
    'CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
}