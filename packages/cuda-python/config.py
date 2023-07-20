
from jetson_containers import L4T_VERSION

if L4T_VERSION.major >= 34:  # JetPack 5
    package['build_args'] = {'CUDA_PYTHON_VERSION': 'v11.7.1'}  # final version before CUDA 12 required
else:
    package = None
