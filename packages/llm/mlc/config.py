
from jetson_containers import CUDA_ARCHITECTURES

package['build_args'] = {
    'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
}

dev_package = package.copy()

dev_package['name'] = 'mlc:dev'
dev_package['dockerfile'] = 'Dockerfile.dev'

package = [package, dev_package]