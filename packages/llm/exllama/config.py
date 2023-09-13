
from jetson_containers import CUDA_ARCHITECTURES

package['build_args'] = {
    'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
}

exllama_v2 = package.copy()

exllama_v2['name'] = 'exllama:v2'
exllama_v2['dockerfile'] = 'Dockerfile.v2'
exllama_v2['test'] = 'test_v2.sh'

package = [package, exllama_v2]