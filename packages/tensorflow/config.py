
from jetson_containers import L4T_VERSION

if L4T_VERSION.major >= 35:   # JetPack 5.0.2 / 5.1.x
    TENSORFLOW1_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v51/tensorflow/tensorflow-1.15.5+nv23.03-cp38-cp38-linux_aarch64.whl'
    TENSORFLOW1_WHL = 'tensorflow-1.15.5+nv23.03-cp38-cp38-linux_aarch64.whl'
    TENSORFLOW2_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v51/tensorflow/tensorflow-2.11.0+nv23.03-cp38-cp38-linux_aarch64.whl'
    TENSORFLOW2_WHL = 'tensorflow-2.11.0+nv23.03-cp38-cp38-linux_aarch64.whl'
elif L4T_VERSION.major == 34:  # JetPack 5.0 / 5.0.1
    TENSORFLOW1_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/tensorflow-1.15.5+nv22.4-cp38-cp38-linux_aarch64.whl'
    TENSORFLOW1_WHL = 'tensorflow-1.15.5+nv22.4-cp38-cp38-linux_aarch64.whl'
    TENSORFLOW2_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/tensorflow-2.8.0+nv22.4-cp38-cp38-linux_aarch64.whl'
    TENSORFLOW2_WHL = 'tensorflow-2.8.0+nv22.4-cp38-cp38-linux_aarch64.whl'
elif L4T_VERSION.major == 32:  # JetPack 4
    TENSORFLOW1_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-1.15.5+nv22.1-cp36-cp36m-linux_aarch64.whl'
    TENSORFLOW1_WHL = 'tensorflow-1.15.5+nv22.1-cp36-cp36m-linux_aarch64.whl'
    TENSORFLOW2_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl'
    TENSORFLOW2_WHL = 'tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl'

# duplicate the packages for separate tf1/tf2 containers
tf1 = package.copy()
tf2 = package.copy()

tf1['build_args'] = {
    'TENSORFLOW_URL': TENSORFLOW1_URL,
    'TENSORFLOW_WHL': TENSORFLOW1_WHL
}

tf2['build_args'] = {
    'TENSORFLOW_URL': TENSORFLOW2_URL,
    'TENSORFLOW_WHL': TENSORFLOW2_WHL
}

package = {'tensorflow': tf1, 'tensorflow2': tf2}

'''
# this way works too
tf1.update({'name': 'tensorflow', 'depends': 'protobuf:cpp'})
tf2.update({'name': 'tensorflow2', 'depends': 'protobuf:cpp'})

package = [tf1, tf2]
'''

