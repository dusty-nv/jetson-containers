from jetson_containers import PYTHON_VERSION, find_container
from packaging.version import Version


builder = package.copy()
runtime = package.copy()

builder['name'] = 'pycuda:builder'
builder['dockerfile'] = 'Dockerfile.builder'
builder['build_args'] = {
    # v2022.1 is the last version to support Python 3.6
    'PYCUDA_VERSION': 'v2022.1' if PYTHON_VERSION == Version('3.6') else 'main',
}

runtime['build_args'] = {
    'BUILD_IMAGE': find_container(builder['name']),
}

package = [builder, runtime]
