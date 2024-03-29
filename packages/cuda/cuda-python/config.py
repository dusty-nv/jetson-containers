
from jetson_containers import L4T_VERSION, find_container

def cuda_python(version, cuda=None, default=False):
    pkg = package.copy()
    
    pkg['name'] = f"cuda-python:{version}"

    if default:
        pkg['alias'] = 'cuda-python'
        
    if not cuda:
        cuda = version
        
    if len(cuda.split('.')) > 2:
        cuda = cuda[:-2]
        
    pkg['depends'] = [f"cuda:{cuda}" if x == 'cuda' else x for x in pkg['depends']]
    
    if len(version.split('.')) < 3:
        version = version + '.0'
        
    pkg['build_args'] = {'CUDA_PYTHON_VERSION': version}
        
    return pkg
    
if L4T_VERSION.major <= 32:
    package = None
else:
    if L4T_VERSION.major >= 36:    # JetPack 6
        package = [
            cuda_python('12.2', default=True),
            cuda_python('12.4')
        ]
    elif L4T_VERSION.major >= 34:  # JetPack 5
        package = [
            cuda_python('11.4', default=True),
            #cuda_python('11.7', '11.4'),
        ]
