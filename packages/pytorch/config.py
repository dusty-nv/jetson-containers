
from jetson_containers import L4T_VERSION


def pytorch_config(version, whl, url, default=False):
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

    return pkg
    

if L4T_VERSION.major >= 35:   # JetPack 5.0.2 / 5.1.x
    configs = [
        ('2.1', 'torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl'),
        ('2.0', 'torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl', True),
        ('1.13', 'torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl'),
        ('1.12', 'torch-1.12.0a0+8a1a93a9.nv22.5-cp38-cp38-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+8a1a93a9.nv22.5-cp38-cp38-linux_aarch64.whl'),
        ('1.11', 'torch-1.11.0-cp38-cp38-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl'),
    ]
elif L4T_VERSION.major == 34:  # JetPack 5.0 / 5.0.1
    configs = [
        ('1.11', 'torch-1.11.0-cp38-cp38-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl', True)
    ]
elif L4T_VERSION.major == 32:  # JetPack 4
    configs = [
        ('1.11', 'torch-1.11.0a0+17540c5-cp36-cp36m-linux_aarch64.whl', 'https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5-cp36-cp36m-linux_aarch64.whl'),
        ('1.10', 'torch-1.10.0-cp36-cp36m-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl', True),  
        ('1.9', 'torch-1.9.0-cp36-cp36m-linux_aarch64.whl', 'https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl'),      
    ]

package = [pytorch_config(c[0], c[1], c[2], c[3] if len(c) > 3 else False) for c in configs]