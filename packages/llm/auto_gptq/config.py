
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

if L4T_VERSION.major >= 36:
    AUTOGPTQ_BRANCH='main'
else:
    AUTOGPTQ_BRANCH='v0.4.2'  # 8/28/2023 - build errors in main (v0.4.2 is tag prior)
    
package['build_args'] = {
    'AUTOGPTQ_BRANCH': AUTOGPTQ_BRANCH,
    'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
}
