
from jetson_containers import CUDA_ARCHITECTURES, CUDA_VERSION, JETPACK_VERSION

def ollama(branch, golang='1.22.8', cmake='3.22.1', requires=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'ollama:{branch}'

    if branch[0].isnumeric():
        branch = 'v' + branch
        
    pkg['build_args'] = {
        'OLLAMA_REPO': 'ollama/ollama',
        'OLLAMA_BRANCH': branch,
        'GOLANG_VERSION': golang,
        'CMAKE_VERSION': cmake,
        'JETPACK_VERSION': str(JETPACK_VERSION),
        'CUDA_VERSION_MAJOR' : CUDA_VERSION.major,
        'CMAKE_CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    }   
     
    if default:
        pkg['alias'] = 'ollama'

    return pkg
    
package = [
    ollama('main'),
    ollama('0.5.5', default=True),
]
