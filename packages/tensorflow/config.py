from jetson_containers import L4T_VERSION, PYTHON_VERSION
from packaging.version import Version

package = []

def tensorflow(version, tensorflow_version='tf2', requires=None, default=False):
    pkg = {}
  
    if default:
        pkg['alias'] = 'tensorflow2' if tensorflow_version == 'tf2' else 'tensorflow1'
        
    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'tensorflow{"" if tensorflow_version == "2" else "1"}:{version}'
    pkg['notes'] = f"TensorFlow {tensorflow_version.upper()} version {version}"
    
    prebuilt_wheels = {
        # TensorFlow tf1
        ('36', '1.15.5', 'tf1'): (None, None),
        ('35', '1.15.5', 'tf1'): (
            'https://developer.download.nvidia.com/compute/redist/jp/v51/tensorflow/tensorflow-1.15.5+nv23.03-cp38-cp38-linux_aarch64.whl',
            'tensorflow-1.15.5+nv23.03-cp38-cp38-linux_aarch64.whl'
        ),
        ('34', '1.15.5', 'tf1'): (
            'https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/tensorflow-1.15.5+nv22.4-cp38-cp38-linux_aarch64.whl',
            'tensorflow-1.15.5+nv22.4-cp38-cp38-linux_aarch64.whl'
        ),
        ('32', '1.15.5', 'tf1'): (
            'https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-1.15.5+nv22.1-cp36-cp36m-linux_aarch64.whl',
            'tensorflow-1.15.5+nv22.1-cp36-cp36m-linux_aarch64.whl'
        ),
        # TensorFlow v2
        ('36', '2.16.1', 'tf2'): (
            'https://developer.download.nvidia.com/compute/redist/jp/v60/tensorflow/tensorflow-2.16.1+nv24.06-cp310-cp310-linux_aarch64.whl',
            'tensorflow-2.16.1+nv24.06-cp310-cp310-linux_aarch64.whl'
        ),
        ('36', '2.17.1', 'tf2'): (
            'https://developer.download.nvidia.com/compute/redist/jp/v61/tensorflow/tensorflow-2.17.1+nv24.07-cp310-cp310-linux_aarch64.whl',
            'tensorflow-2.17.1+nv24.07-cp310-cp310-linux_aarch64.whl'
        ),
        ('35', '2.15.0', 'tf2'): (
            'https://developer.download.nvidia.com/compute/redist/jp/v51/tensorflow/tensorflow-2.15.0+nv23.05-cp38-cp38-linux_aarch64.whl',
            'tensorflow-2.15.0+nv23.05-cp38-cp38-linux_aarch64.whl'
        ),
        ('35', '2.11.0', 'tf2'): (
            'https://developer.download.nvidia.com/compute/redist/jp/v51/tensorflow/tensorflow-2.11.0+nv23.03-cp38-cp38-linux_aarch64.whl',
            'tensorflow-2.11.0+nv23.03-cp38-cp38-linux_aarch64.whl'
        ),
        ('34', '2.8.0', 'tf2'): (
            'https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/tensorflow-2.8.0+nv22.4-cp38-cp38-linux_aarch64.whl',
            'tensorflow-2.8.0+nv22.4-cp38-cp38-linux_aarch64.whl'
        ),
        ('32', '2.7.0', 'tf2'): (
            'https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl',
            'tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl'
        ),
        # Puedes agregar más entradas si hay ruedas precompiladas disponibles
    }
    
    L4T_MAJOR = str(L4T_VERSION.major)
    wheel_key = (L4T_MAJOR, version, tensorflow_version)
    
    if wheel_key in prebuilt_wheels and prebuilt_wheels[wheel_key][0] is not None:
        url, whl = prebuilt_wheels[wheel_key]
        pkg['build_args'] = {
            'TENSORFLOW_VERSION': version,
            'TENSORFLOW_URL': url,
            'TENSORFLOW_WHL': whl,
            'PYTHON_VERSION_MAJOR': PYTHON_VERSION.major,
            'PYTHON_VERSION_MINOR': PYTHON_VERSION.minor,
            'BUILD_FROM_SOURCE': 'off'  # Usar rueda precompilada
        }
    else:
        # No hay rueda precompilada disponible, configurar para construir desde el código fuente
        pkg['build_args'] = {
            'TENSORFLOW_VERSION': version,
            'PYTHON_VERSION_MAJOR': PYTHON_VERSION.major,
            'PYTHON_VERSION_MINOR': PYTHON_VERSION.minor,
            'BUILD_FROM_SOURCE': 'on',  # Construir desde el código fuente
            'TENSORFLOW_VERSION_TAG': tensorflow_version  # 'tf1' o 'tf2'
        }
        pkg['notes'] += " (will be built from source)"
    
    builder = pkg.copy()
    builder['name'] = f'{pkg["name"]}-builder'
    builder['build_args'] = {**pkg['build_args'], 'FORCE_BUILD': 'on'}

    return [pkg, builder]

package = [
    # TensorFlow tf1
    *tensorflow(
        version='1.15.5',
        tensorflow_version='tf1',
        default=(L4T_VERSION.major == 35),
        requires='>=32,<36'
    ),
    # TensorFlow tf2 para L4T >=36
    *tensorflow(
        version='2.16.1',
        tensorflow_version='tf2',
        default=(L4T_VERSION.major >= 36),
        requires='>=36'
    ),
    *tensorflow(
        version='2.17.1',
        tensorflow_version='tf2',
        requires='>=36'
    ),
]
