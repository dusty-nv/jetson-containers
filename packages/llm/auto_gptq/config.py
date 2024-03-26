from jetson_containers import CUDA_ARCHITECTURES, find_container

"""
if L4T_VERSION.major >= 36:
    AUTOGPTQ_BRANCH='main'
else:
    AUTOGPTQ_BRANCH='v0.4.2'  # 8/28/2023 - build errors in main (v0.4.2 is tag prior)
"""

AUTOGPTQ_BRANCH = 'main'

builder = package.copy()
runtime = package.copy()

builder['name'] = 'auto_gptq:builder'
builder['dockerfile'] = 'Dockerfile.builder'

builder['build_args'] = {
    'AUTOGPTQ_BRANCH': AUTOGPTQ_BRANCH,
    'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
}

runtime['build_args'] = {
    'BUILD_IMAGE': find_container(builder['name']),
}

package = [builder, runtime]
