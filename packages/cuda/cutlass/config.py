
from jetson_containers import L4T_VERSION, CUDA_VERSION, update_dependencies
from packaging.version import Version

def cutlass(version, cuda=None):
    pkg = package.copy()

    pkg['name'] = f"cuda-python:{version}"

    if not cuda:
        cuda = version

    if len(cuda.split('.')) > 2:
        cuda = cuda[:-2]

    pkg['depends'] = update_dependencies(pkg['depends'], f"cuda:{cuda}")

    if len(version.split('.')) < 3:
        version = version + '.0'

    pkg['build_args'] = {'CUTLASS_VERSION': version}

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if Version(version) == CUDA_VERSION:
        pkg['alias'] = 'cuda-python'
        builder['alias'] = 'cuda-python:builder'

    return pkg, builder

if L4T_VERSION.major >= 36:    # JetPack 6
    package = [
        cutlass('4.0.0'),
    ]
