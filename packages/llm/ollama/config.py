import os

from jetson_containers import (
    CUDA_ARCHITECTURES, CUDA_VERSION, 
    JETPACK_VERSION, nv_tegra_release
)

nv_tegra_release( # ollama uses /etc/nv_tegra_release
    dst=os.path.join(package['path'], 'nv_tegra_release')
) 

def ollama(version, golang='1.22.8', cmake='3.22.1', branch=None, requires=None, default=False):
    pkg = package.copy()

    if not branch:
        branch = version

    pkg['name'] = f'ollama:{version}'

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
    ollama('0.4.0'),
    ollama('0.5.1'),
    ollama('0.5.5', branch='0.5.5-rc0'),
    ollama('0.5.7'),
    ollama('0.6.3', default=True),
]
