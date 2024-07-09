
from jetson_containers import L4T_VERSION

if L4T_VERSION.major >= 36:    # JetPack 6.0
    TENSORFLOW1_URL = None
    TENSORFLOW1_WHL = None
    TENSORFLOW2_URL = 'https://developer.download.nvidia.com/compute/redist/jp/v60/tensorflow/tensorflow-2.16.1+nv24.06-cp310-cp310-linux_aarch64.whl' # 'https://nvidia.box.com/shared/static/wp43cd8e0lgen2wdqic3irdwagpgn0iz.whl'
    TENSORFLOW2_WHL = 'tensorflow-2.16.1+nv24.06-cp310-cp310-linux_aarch64.whl' #'tensorflow-2.14.0+nv23.11-cp310-cp310-linux_aarch64.whl'
elif L4T_VERSION.major == 35:  # JetPack 5.0.2 / 5.1.x
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

# package templates for separate tf1/tf2 containers
tf_pack = package
package = {}

if TENSORFLOW1_WHL:
    tf1 = tf_pack.copy()
    
    tf1['build_args'] = {
        'TENSORFLOW_URL': TENSORFLOW1_URL,
        'TENSORFLOW_WHL': TENSORFLOW1_WHL
    }
    
    package['tensorflow'] = tf1
    
if TENSORFLOW2_WHL:
    tf2 = tf_pack.copy()

    tf2['build_args'] = {
        'TENSORFLOW_URL': TENSORFLOW2_URL,
        'TENSORFLOW_WHL': TENSORFLOW2_WHL
    }

    package['tensorflow2'] = tf2
    
