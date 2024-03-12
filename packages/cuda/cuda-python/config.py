
from jetson_containers import L4T_VERSION, find_container

if L4T_VERSION.major <= 32:
    package = None
else:
    builder = package.copy()
    runtime = package.copy()
    
    if L4T_VERSION.major >= 36:    # JetPack 6
        builder['build_args'] = {'CUDA_PYTHON_VERSION': 'v12.2.0'} 
    elif L4T_VERSION.major >= 34:  # JetPack 5
        builder['build_args'] = {'CUDA_PYTHON_VERSION': 'v11.7.0'}  # final version before CUDA 12 required

    builder['name'] = 'cuda-python:builder'
    builder['dockerfile'] = 'Dockerfile.builder'
    
    runtime['build_args'] = {
        'BUILD_IMAGE': find_container(builder['name']),
    }
    
    package = [builder, runtime]
