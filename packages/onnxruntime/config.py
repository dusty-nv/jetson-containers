
from jetson_containers import L4T_VERSION

# on JetPack 4, newer ORT builds fail due to gcc7
if L4T_VERSION.major <= 32:
    package['build_args'] = {'ONNXRUNTIME_VERSION': 'v1.15.1'}
    
