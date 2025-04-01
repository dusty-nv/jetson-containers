
from jetson_containers import CUDA_ARCHITECTURES

package['build_args'] = {
    'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES]),
}
