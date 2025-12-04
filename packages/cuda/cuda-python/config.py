from jetson_containers import L4T_VERSION, CUDA_VERSION, update_dependencies
from packaging.version import Version

def _build_version_string(v: Version) -> str:
    """
    Return version for build arg:
      - If only MAJOR.MINOR were given (no pre/dev/post/local), force x.y.0
      - Otherwise, use canonical PEP 440 string (preserves .postN, aN, bN, rcN, .devN)
    Examples:
      "12.6"         -> "12.6.0"
      "12.6.post1"   -> "12.6.post1"
      "13.0.1"       -> "13.0.1"
      "13.0rc2"      -> "13.0rc2"
    """
    # v.public is canonical (e.g., "12.6.0.post1" -> "12.6.post1")
    if len(v.release) == 2 and not (v.pre or v.post or v.dev or v.local):
        return f"{v.major}.{v.minor}.0"
    return v.public

def cuda_python(version, cuda=None):
    pkg = package.copy()
    pkg['name'] = f"cuda-python:{version}"

    v = Version(str(version))

    # If 'cuda' not given, depend on the same MAJOR.MINOR as 'version'
    if cuda is None:
        cuda_major_minor = f"{v.major}.{v.minor}"
    else:
        cv = Version(str(cuda))
        cuda_major_minor = f"{cv.major}.{cv.minor}"

    # Depend only on CUDA MAJOR.MINOR
    pkg['depends'] = update_dependencies(pkg['depends'], f"cuda:{cuda_major_minor}")

    # Build arg should reflect a full version when appropriate, and preserve post/dev/pre
    version_norm = _build_version_string(v)
    pkg['build_args'] = {'CUDA_PYTHON_VERSION': version_norm}

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    # Consider equal if MAJOR.MINOR match, regardless of patch/post
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
            cuda_python('12.6.2.post1'),
            cuda_python('12.8'),
            cuda_python('12.9'),
            cuda_python('13.0.3'),
            cuda_python('13.1'),
            # Please enable only when added as package in cuda config.py
            # cuda_python('13.1'),
        ]
    elif L4T_VERSION.major >= 34:  # JetPack 5
        package = [
            cuda_python('11.4'),
            #cuda_python('11.7', '11.4'),
        ]
