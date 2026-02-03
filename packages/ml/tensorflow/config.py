from jetson_containers import L4T_VERSION, PYTHON_VERSION, CUDA_VERSION, IS_SBSA
from packaging.version import Version

from ..ml.tensorflow.version import TENSORFLOW_VERSION
from ..cuda.cudastack.config import CUDNN_VERSION


def tensorflow(version, tensorflow_version='tf2', requires=None, default=False):
    pkg = package.copy()

    if default:
        pkg['alias'] = ['tensorflow2'] if tensorflow_version == 'tf2' else ['tensorflow1']

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'tensorflow{"2" if tensorflow_version == "tf2" else "1"}:{version}'
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
            'FORCE_BUILD': 'off',
            'IS_SBSA': int(IS_SBSA)
        }
        pkg['dockerfile'] = 'Dockerfile'
    else:
        pkg['build_args'] = {
            'TENSORFLOW_VERSION': version,
            'PYTHON_VERSION_MAJOR': PYTHON_VERSION.major,
            'PYTHON_VERSION_MINOR': PYTHON_VERSION.minor,
            'CUDA_VERSION_MAJOR': CUDA_VERSION.major,
            'CUDA_VERSION_MINOR': CUDA_VERSION.minor,
            'CUDNN_VERSION_MAJOR': CUDNN_VERSION.major,
            'CUDNN_VERSION_MINOR': CUDNN_VERSION.minor,
            'FORCE_BUILD': 'off',
            'IS_SBSA': int(IS_SBSA)
        }
        pkg['notes'] += " (will be built from source)"
        pkg['dockerfile'] = 'Dockerfile.pip'
        pkg['alias'] = [f'tensorflow2:{version}' if tensorflow_version == 'tf2' else f'tensorflow1:{version}']

    builder = pkg.copy()
    builder['name'] = f'{pkg["name"]}-builder'
    builder['build_args'] = {**pkg['build_args'], 'FORCE_BUILD': 'on'}
    builder['alias'] = [f'tensorflow2:{version}-builder' if tensorflow_version == 'tf2' else f'tensorflow1:{version}-builder']

    if Version(version) == TENSORFLOW_VERSION:
        pkg['alias'].append('tensorflow')
        builder['alias'].append('tensorflow:builder')

        if tensorflow_version == 'tf2':
            pkg['alias'].append('tensorflow2')
            builder['alias'].append('tensorflow2:builder')

    return [pkg, builder]

package = [
    # TensorFlow tf1
    *tensorflow(
        version='1.15.5',
        tensorflow_version='tf1',
        default=(L4T_VERSION.major <= 35),
        requires='<36'
    ),
*tensorflow(
        version='2.21.0',
        tensorflow_version='tf2',
        requires='>=36',
        default=(CUDA_VERSION >= Version('12.6')), # Blackwell Support
    ),
]
