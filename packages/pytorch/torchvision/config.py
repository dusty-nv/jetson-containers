
from jetson_containers import L4T_VERSION

if L4T_VERSION.major >= 35:    # JetPack 5.0.2 / 5.1.x
    TORCHVISION_VERSION = 'v0.15.1'
    TORCH_CUDA_ARCH_LIST = "7.2;8.7"
elif L4T_VERSION.major == 34:  # JetPack 5.0 / 5.0.1
    TORCHVISION_VERSION = 'v0.12.0'
    TORCH_CUDA_ARCH_LIST = '7.2;8.7'
elif L4T_VERSION.major == 32:  # JetPack 4
    TORCHVISION_VERSION = 'v0.11.1'
    TORCH_CUDA_ARCH_LIST = "5.3;6.2;7.2"

package['build_args'] = {
    'TORCHVISION_VERSION': TORCHVISION_VERSION,
    'TORCH_CUDA_ARCH_LIST': TORCH_CUDA_ARCH_LIST
}

package['depends'] = 'pytorch'
