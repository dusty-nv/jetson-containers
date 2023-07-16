
from jetson_containers import L4T_VERSION, PYTHON_VERSION
from packaging.version import Version

if L4T_VERSION >= Version('35.2.1'): # JetPack 5.1
    DEEPSTREAM_URL = 'https://nvidia.box.com/shared/static/2q5nnsvj6bdyk4qir4uglfyuc7t55m7e.tbz2'
    DEEPSTREAM_TAR = 'deepstream_sdk_v6.2.0_jetson.tbz2'
elif L4T_VERSION >= Version('34'):  # JetPack 5.0
    DEEPSTREAM_URL = 'https://nvidia.box.com/shared/static/8jdbxu016wrjz8g5q7dzetj2seksmih9.tbz2'
    DEEPSTREAM_TAR = 'deepstream_sdk_v6.1.0_jetson.tbz2'
elif L4T_VERSION >= Version('32.6'): # JetPack 4.6
    DEEPSTREAM_URL = 'https://nvidia.box.com/shared/static/uetg0vwukwuqf8109i5coq0rizuflniu.tbz2'
    DEEPSTREAM_TAR = 'deepstream_sdk_v6.0.0_jetson.tbz2'
else:
    package = None
    
if package:
    package['build_args'] = {
        'DEEPSTREAM_URL': DEEPSTREAM_URL,
        'DEEPSTREAM_TAR': DEEPSTREAM_TAR,
        'PYTHON_VERSION_MAJOR': PYTHON_VERSION.major,
        'PYTHON_VERSION_MINOR': PYTHON_VERSION.minor,
    }
