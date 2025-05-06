import os

from jetson_containers import JETPACK_VERSION, nv_tegra_release

nv_tegra_release( # ollama uses /etc/nv_tegra_release
    dst=os.path.join(package['path'], 'nv_tegra_release')
) 

def ollama(version, default=False):
    pkg = package.copy()

    pkg['name'] = f'ollama:{version}'

    pkg['build_args'] = {
        'OLLAMA_VERSION': version,
        'JETPACK_VERSION_MAJOR': JETPACK_VERSION.major,
    }   
     
    if default:
        pkg['alias'] = 'ollama'

    return pkg
    
package = [
    #ollama('main'),
    ollama('0.4.0'),
    ollama('0.5.1'),
    ollama('0.5.5'), #, branch='0.5.5-rc0'),
    ollama('0.5.7'),
    ollama('0.6.7'),
    ollama('0.6.8', default=True)
]
