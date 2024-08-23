
from jetson_containers import L4T_VERSION, CUDA_VERSION, update_dependencies
from packaging.version import Version

def cuda_python(version, cuda=None):
    pkg = package.copy()
    
    pkg['name'] = f"cuda-python:{version}"

    if Version(version) == CUDA_VERSION:
        pkg['alias'] = 'cuda-python'
        
    if not cuda:
        cuda = version
        
    if len(cuda.split('.')) > 2:
        cuda = cuda[:-2]
        
    pkg['depends'] = update_dependencies(pkg['depends'], f"cuda:{cuda}")
    
    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {'CUDA_PYTHON_VERSION': version}
        
    return pkg
    
if L4T_VERSION.major <= 32:
    package = None
else:
    if L4T_VERSION.major >= 36:    # JetPack 6
        package = [
            cuda_python('12.2'),
            cuda_python('12.4'),
            cuda_python('12.6'),
        ]
    elif L4T_VERSION.major >= 34:  # JetPack 5
        package = [
            cuda_python('11.4'),
            #cuda_python('11.7', '11.4'),
        ]
