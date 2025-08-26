
from jetson_containers import L4T_VERSION, CUDA_VERSION, update_dependencies
from packaging.version import Version

def cuda_python(version, cuda=None):
    pkg = package.copy()
    pkg['name'] = f"cuda-python:{version}"

    v = Version(version)

    # If 'cuda' not given, depend on the same MAJOR.MINOR as 'version'
    if cuda is None:
        cuda_major_minor = f"{v.major}.{v.minor}"
    else:
        cv = Version(str(cuda))
        cuda_major_minor = f"{cv.major}.{cv.minor}"

    # Depend on CUDA MAJOR.MINOR (not patch), robustly derived
    pkg['depends'] = update_dependencies(pkg['depends'], f"cuda:{cuda_major_minor}")

    # Build arg should always be a full x.y.z
    version_norm = f"{v.major}.{v.minor}.{v.micro}"
    pkg['build_args'] = {'CUDA_PYTHON_VERSION': version_norm}

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    # Consider equal if MAJOR.MINOR match, regardless of patch
    if (v.major, v.minor) == (CUDA_VERSION.major, CUDA_VERSION.minor):
        pkg['alias'] = 'cuda-python'
        builder['alias'] = 'cuda-python:builder'

    return pkg, builder

if L4T_VERSION.major <= 32:
    package = None
else:
    if L4T_VERSION.major >= 36:    # JetPack 6
        package = [
            cuda_python('12.2'),
            cuda_python('12.4'),
            cuda_python('12.6'),
            cuda_python('12.8'),
            cuda_python('12.9'),
            cuda_python('13.0'),
            # Please enable only when added as package in cuda config.py
            # cuda_python('13.1'),
        ]
    elif L4T_VERSION.major >= 34:  # JetPack 5
        package = [
            cuda_python('11.4'),
            #cuda_python('11.7', '11.4'),
        ]
