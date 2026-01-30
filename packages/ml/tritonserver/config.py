import os

from jetson_containers import L4T_VERSION, LSB_RELEASE
from packaging.version import Version

TRITON_CLIENTS = 'clients'

if L4T_VERSION >= Version('36.4.0'): # JetPack 6.2.1 DP
    # https://github.com/triton-inference-server/server/releases/tag/v2.49.0
    if LSB_RELEASE == '22.04':
        TRITON_VERSION = '2.49.0'
        TRITON_URL = f'https://github.com/triton-inference-server/server/releases/download/v{TRITON_VERSION}/tritonserver{TRITON_VERSION}-igpu.tar.gz'
        TRITON_TAR = f'tritonserver{TRITON_VERSION}-igpu.tar.gz'
    else:
        TRITON_VERSION = '2.63.0'
        TRITON_URL = f'https://github.com/triton-inference-server/server/releases/download/v{TRITON_VERSION}/tritonserver{TRITON_VERSION}-igpu.tar.gz'
        TRITON_TAR = f'tritonserver{TRITON_VERSION}-igpu.tar.gz'
    
    TRITON_CLIENTS = 'tritonserver/clients'
elif L4T_VERSION >= Version('36.0.0'): # JetPack 6.0 DP
    # https://github.com/triton-inference-server/server/releases/tag/v2.42.0
    TRITON_URL = 'https://github.com/triton-inference-server/server/releases/download/v2.59.1/tritonserver2.59.1-igpu.tar'
    TRITON_TAR = 'tritonserver2.59.1-igpu.tar'
    TRITON_CLIENTS = 'tritonserver/clients'
elif L4T_VERSION >= Version('35.3.1'): # JetPack 5.1.1
    # https://github.com/triton-inference-server/server/releases/tag/v2.35.0
    TRITON_URL = 'https://github.com/triton-inference-server/server/releases/download/v2.35.0/tritonserver2.35.0-jetpack5.1.2-update-1.tgz'
    TRITON_TAR = 'tritonserver2.35.0-jetpack5.1.2-update-1.tgz'
    TRITON_CLIENTS = 'tritonserver/clients'  # in 2.35, clients/ dir moved under tritonserver/
elif L4T_VERSION >= Version('35.2.1'): # JetPack 5.1
    # https://github.com/triton-inference-server/server/releases/tag/v2.34.0
    TRITON_URL = 'https://github.com/triton-inference-server/server/releases/download/v2.34.0/tritonserver2.34.0-jetpack5.1.tgz'
    TRITON_TAR = 'tritonserver2.34.0-jetpack5.1.tgz'
elif L4T_VERSION >= Version('35'): # JetPack 5.0.2 / L4T R35.1
    # https://github.com/triton-inference-server/server/releases/tag/v2.27.0
    TRITON_URL = 'https://github.com/triton-inference-server/server/releases/download/v2.27.0/tritonserver2.27.0-jetpack5.0.2.tgz'
    TRITON_TAR = 'tritonserver2.27.0-jetpack5.0.2.tgz'
elif L4T_VERSION >= Version('34'): # JetPack 5.0/5.0.1 DP
    # https://github.com/triton-inference-server/server/releases/tag/v2.23.0
    TRITON_URL = 'https://github.com/triton-inference-server/server/releases/download/v2.23.0/tritonserver2.23.0-jetpack5.0.tgz'
    TRITON_TAR = 'tritonserver2.23.0-jetpack5.0.tgz'
elif L4T_VERSION >= Version('32.7'): # JetPack 4.6.1+
    # https://github.com/triton-inference-server/server/releases/tag/v2.19.0
    TRITON_URL = 'https://github.com/triton-inference-server/server/releases/download/v2.19.0/tritonserver2.19.0-jetpack4.6.1.tgz'
    TRITON_TAR = 'tritonserver2.19.0-jetpack4.6.1.tgz'
elif L4T_VERSION >= Version('32.6'): # JetPack 4.6
    # https://github.com/triton-inference-server/server/releases/tag/v2.17.0
    TRITON_URL = 'https://github.com/triton-inference-server/server/releases/download/v2.17.0/tritonserver2.17.0-jetpack4.6.tgz'
    TRITON_TAR = 'tritonserver2.17.0-jetpack4.6.tgz'
else:
    print('-- tritonserver not available before JetPack 4.6 / L4T R32.6')
    package = None

if package:
    package['build_args'] = {
        'TRITON_URL': TRITON_URL,
        'TRITON_TAR': TRITON_TAR,
        'TRITON_VERSION': os.path.basename(os.path.dirname(TRITON_URL)).lstrip('v'),
        'TRITON_CLIENTS': TRITON_CLIENTS,
    }
