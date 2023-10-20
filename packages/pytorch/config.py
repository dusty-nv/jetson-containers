
from jetson_containers import L4T_VERSION
import re

def pytorch(version, whl, url, requires, default=False):
    """
    Create a version of PyTorch for the package list
    """
    pkg = package.copy()
    
    pkg['name'] = f'pytorch:{version}'
    pkg['alias'] = [f'torch:{version}']
    
    if default:
        pkg['alias'].extend(['pytorch', 'torch'])
        
    pkg['build_args'] = {
        'PYTORCH_WHL': whl,
        'PYTORCH_URL': url,
    }

    pkg['requires'] = requires
    
    return pkg


def pytorch_source(version, dockerfile, build_env_variables, requires, default=False):
    """
    Create a version of PyTorch for the package list
    """
    pkg = package.copy()
    
    name_suffix = ''
    for env_var in build_env_variables.split():
        print(f'################### env_var: {env_var}')
        if bool(re.match('^USE_(.+)=1', env_var)):
            print(f'################### Matched ^USE_(.+)=1 pattern')
            m = re.match('^USE_(.+)=1', env_var)
            name_suffix=f'{name_suffix}-{m.group(1).lower()}'
        elif bool(re.match('^USE_(.+)=0', env_var)):
            print(f'################### Matched ^USE_(.+)=0 pattern')
            m = re.match('^USE_(.+)=0', env_var)
            name_suffix=f'{name_suffix}-no-{m.group(1).lower()}'
        else:
            print(f'################### No match')
            name_suffix=f"{name_suffix}-{env_var.replace('=', '-')}"
    print(f'################### name_suffix: {name_suffix}')

    pkg['name'] = f'pytorch:{version}{name_suffix}'
    pkg['alias'] = [f'torch:{version}{name_suffix}']
    
    if default:
        pkg['alias'].extend(['pytorch', 'torch'])

    pkg['build_args'] = {
        'PYTORCH_BUILD_VERSION': version,
        'PYTORCH_BUILD_NUMBER': '1',
        'PYTORCH_BUILD_EXTRA_ENV': build_env_variables,
    }

    pkg['dockerfile'] = dockerfile

    print(f'######### {requires}')
    pkg['requires'] = requires
    
    return pkg
    
package = [
    # JetPack 5
    pytorch('2.1', 'torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl', '>=34.1.0'),
    pytorch('2.0', 'torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl', '>=34.1.0', default=True),
    pytorch('1.13', 'torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl', '>=34.1.0'),
    pytorch('1.12', 'torch-1.12.0a0+8a1a93a9.nv22.5-cp38-cp38-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+8a1a93a9.nv22.5-cp38-cp38-linux_aarch64.whl', '>=34.1.0'),
    pytorch('1.11', 'torch-1.11.0-cp38-cp38-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl', '>=34.1.0'),
    
    # JetPack 4
    #pytorch('1.11', 'torch-1.11.0a0+17540c5-cp36-cp36m-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp36-cp36m-linux_aarch64.whl', '==32.*'),  # (built without LAPACK support)
    pytorch('1.10', 'torch-1.10.0-cp36-cp36m-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl', '==32.*', default=True),
    pytorch('1.9', 'torch-1.9.0-cp36-cp36m-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl', '==32.*'),

    # Build from source
    pytorch_source('2.1.0', 'Dockerfile.2.x-build', 
                    'USE_DISTRIBUTED=1', 
                    ['python', 'numpy', 'onnx']),
]
