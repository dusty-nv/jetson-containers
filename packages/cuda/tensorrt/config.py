import os

from packaging.version import Version
from jetson_containers import (
    L4T_VERSION, CUDA_VERSION, SYSTEM_ARM,
    update_dependencies, package_requires, IS_TEGRA, IS_SBSA
)

package['depends'] = ['cuda', 'cudnn', 'python']
package['test'] = ['test.sh']

# Define the default TENSORRT_VERSION either from environment variable or
# as to what version of TensorRT was released with that version of CUDA
if 'TENSORRT_VERSION' in os.environ and len(os.environ['TENSORRT_VERSION']) > 0:
    TENSORRT_VERSION = Version(os.environ['TENSORRT_VERSION'])
elif SYSTEM_ARM:
    if IS_SBSA:
        if CUDA_VERSION >= Version('13.0'):
            TENSORRT_VERSION = Version('10.13')
        else:
            TENSORRT_VERSION = Version('10.13')
    else:
        # Tegra
        if L4T_VERSION.major >= 36:
            if CUDA_VERSION >= Version('13.0'):
                TENSORRT_VERSION = Version('10.13')
            elif CUDA_VERSION >= Version('12.9'):
                TENSORRT_VERSION = Version('10.13')
            elif CUDA_VERSION >= Version('12.8'):
                TENSORRT_VERSION = Version('10.7')
            elif CUDA_VERSION >= Version('12.6'):
                TENSORRT_VERSION = Version('10.3')
            elif CUDA_VERSION == Version('12.4'):
                TENSORRT_VERSION = Version('10.0')
            else:
                TENSORRT_VERSION = Version('8.6')
        elif L4T_VERSION.major >= 34:
            TENSORRT_VERSION = Version('8.5')
        elif L4T_VERSION.major >= 32:
            TENSORRT_VERSION = Version('8.2')
else:
    TENSORRT_VERSION = Version('10.13') # x86_64


def tensorrt_deb(version, url, deb, cudnn=None, packages=None, requires=None):
    """
    Generate containers for a particular version of TensorRT installed from debian packages
    """
    if not packages:
        packages = os.environ.get('TENSORRT_PACKAGES', 'tensorrt tensorrt-libs python3-libnvinfer-dev')

    tensorrt = package.copy()

    tensorrt['name'] = f'tensorrt:{version}'
    tensorrt['dockerfile'] = 'Dockerfile.deb'

    tensorrt['build_args'] = {
        'TENSORRT_URL': url,
        'TENSORRT_DEB': deb,
        'TENSORRT_PACKAGES': packages,
    }

    if Version(version) == TENSORRT_VERSION:
        tensorrt['alias'] = 'tensorrt'

    if cudnn:
        tensorrt['depends'] = update_dependencies(tensorrt['depends'], f"cudnn:{cudnn}")

    if requires:
        tensorrt['requires'] = requires

    package_requires(tensorrt, system_arch='aarch64') # default to aarch64

    return tensorrt


def tensorrt_tar(version, url, cudnn=None, requires=None):
    """
    Generate containers for a particular version of TensorRT installed from tar.gz file
    """
    tensorrt = package.copy()

    tensorrt['name'] = f'tensorrt:{version}'

    tensorrt['dockerfile'] = 'Dockerfile.tar'
    tensorrt['build_args'] = {'TENSORRT_URL': url}

    if Version(version) == TENSORRT_VERSION:
        tensorrt['alias'] = 'tensorrt'

    if cudnn:
        tensorrt['depends'] = update_dependencies(tensorrt['depends'], f"cudnn:{cudnn}")

    if requires:
        tensorrt['requires'] = requires

    package_requires(tensorrt, system_arch='aarch64') # default to aarch64

    return tensorrt


def tensorrt_builtin(version=None, requires=None, default=False):
    """
    Backwards-compatability for when TensorRT already installed in base container (like l4t-jetpack)
    """
    passthrough = package.copy()

    if version is not None:
        if not isinstance(version, str):
            version = f'{version.major}.{version.minor}'

        if default:
            passthrough['alias'] = 'tensorrt'

        passthrough['name'] += f':{version}'

    if requires:
        passthrough['requires'] = requires

    #del passthrough['dockerfile']
    return passthrough

TENSORRT_URL='https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt'

if IS_TEGRA:
    package = [
        # JetPack 6.1
        tensorrt_deb('8.6', 'https://nvidia.box.com/shared/static/hmwr57hm88bxqrycvlyma34c3k4c53t9.deb','nv-tensorrt-local-repo-l4t-8.6.2-cuda-12.2', cudnn='8.9', requires=['==r36.*', '==cu122']),
        # tensorrt_tar('9.3', 'https://nvidia.box.com/shared/static/fp3o14iq7qbm67qjuqivdrdch7009axu.gz', cudnn='8.9', requires=['==r36.*', '==cu122']),

        # JetPack 6.1+ with upgraded CUDA
        tensorrt_tar('10.0', f'{TENSORRT_URL}/10.0.1/tars/TensorRT-10.0.1.6.l4t.aarch64-gnu.cuda-12.4.tar.gz', cudnn='9.0', requires=['==r36.*', '==cu124']),
        tensorrt_tar('10.3', f'{TENSORRT_URL}/10.3.0/tars/TensorRT-10.3.0.26.l4t.aarch64-gnu.cuda-12.6.tar.gz', cudnn='9.3', requires=['==r36.*', '==cu126']),
        tensorrt_tar('10.4', f'{TENSORRT_URL}/10.4.0/tars/TensorRT-10.4.0.26.l4t.aarch64-gnu.cuda-12.6.tar.gz', cudnn='9.3', requires=['==r36.*', '==cu126']),
        tensorrt_tar('10.5', f'{TENSORRT_URL}/10.5.0/tars/TensorRT-10.5.0.18.l4t.aarch64-gnu.cuda-12.6.tar.gz', cudnn='9.3', requires=['==r36.*', '==cu126']),
        tensorrt_tar('10.7', f'{TENSORRT_URL}/10.7.0/tars/TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz', cudnn='9.3', requires=['==r36.*', '==cu126']),
        tensorrt_tar('10.7', f'{TENSORRT_URL}/10.7.0/tars/TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz', cudnn='9.8', requires=['==r36.*', '==cu128']),
        tensorrt_tar('10.7', f'{TENSORRT_URL}/10.7.0/tars/TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz', cudnn='9.9', requires=['==r36.*', '==cu129']),
        tensorrt_tar('10.7', f'{TENSORRT_URL}/10.7.0/tars/TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz', cudnn='9.10', requires=['==r36.*', '==cu129']),
        tensorrt_tar('10.13', f'{TENSORRT_URL}/10.13.2/tars/TensorRT-10.13.2.6.Linux.aarch64-gnu.cuda-13.0.tar.gz', cudnn='9.13.0', requires=['==r36.*', '==cu129']),
        # JetPack 4-5 (TensorRT installed in base container)
        tensorrt_builtin(requires='<36', default=True),
    ]

elif IS_SBSA:
    # sbsa
    package = [
        tensorrt_tar('10.9',f'{TENSORRT_URL}/10.9.0/tars/TensorRT-10.9.0.34.Linux.aarch64-gnu.cuda-12.8.tar.gz', cudnn='9.8', requires='aarch64'),
        tensorrt_tar('10.12',f'{TENSORRT_URL}/10.12.0/tars/TensorRT-10.12.0.36.Linux.aarch64-gnu.cuda-12.9.tar.gz', cudnn='9.10', requires='aarch64'),
        tensorrt_tar('10.13', f'{TENSORRT_URL}/10.13.2/tars/TensorRT-10.13.2.6.Linux.aarch64-gnu.cuda-13.0.tar.gz', cudnn='9.12.0', requires=['aarch64']),
    ]

else:
    # x86_64
    package = [
        tensorrt_tar('10.9', f'{TENSORRT_URL}/10.9.0/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz', cudnn='9.8', requires='x86_64'),
        tensorrt_tar('10.12', f'{TENSORRT_URL}/10.12.0/tars/TensorRT-10.12.0.36.Linux.x86_64-gnu.cuda-12.9.tar.gz', cudnn='9.10', requires='x86_64'),
        tensorrt_tar('10.13', f'{TENSORRT_URL}/10.13.2/tars/TensorRT-10.13.2.6.Linux.x86_64-gnu.cuda-13.0.tar.gz', cudnn='9.12.0', requires='x86_64'),
    ]
