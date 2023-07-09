
from jetson_containers import L4T_VERSION

if L4T_VERSION.major >= 35:   # JetPack 5.0.2 / 5.1.x
    PYTORCH_URL = 'https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl'
    PYTORCH_WHL = 'torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl'
elif L4T_VERSION.major == 34:  # JetPack 5.0 / 5.0.1
    PYTORCH_URL = 'https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl'
    PYTORCH_WHL = 'torch-1.11.0-cp38-cp38-linux_aarch64.whl'
elif L4T_VERSION.major == 32:  # JetPack 4
    PYTORCH_URL = 'https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl'
    PYTORCH_WHL = 'torch-1.10.0-cp36-cp36m-linux_aarch64.whl'

package['build_args'] = {
    'PYTORCH_URL': PYTORCH_URL,
    'PYTORCH_WHL': PYTORCH_WHL
}

package['alias'] = 'torch'
package['depends'] = ['python', 'numpy']
package['category'] = 'ml'
