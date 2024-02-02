
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

package['build_args'] = {
    'AUTOAWQ_BRANCH': 'main',
    'AUTOAWQ_CUDA_ARCH': ','.join([str(x) for x in CUDA_ARCHITECTURES])
}
