
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

if L4T_VERSION.major >= 35:    # JetPack 5.0.2 / 5.1.x
    TORCHVISION_VERSION = 'v0.15.1'
elif L4T_VERSION.major == 34:  # JetPack 5.0 / 5.0.1
    TORCHVISION_VERSION = 'v0.12.0'
elif L4T_VERSION.major == 32:  # JetPack 4
    TORCHVISION_VERSION = 'v0.11.1'

package['build_args'] = {
    'TORCHVISION_VERSION': TORCHVISION_VERSION,
    'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
}

package['depends'] = ['cmake', 'pytorch']
package['category'] = 'ml'
