
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

package['build_args'] = {
    'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
}

if L4T_VERSION.major <= 32:
    package['dockerfile'] = 'Dockerfile.jp4'
    package['depends'].extend(['rust', 'protobuf:apt'])
elif L4T_VERSION.major <= 35:
    package['build_args'] = {
        'LD_PRELOAD_LIBS': '/usr/local/lib/python3.8/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0'
    }