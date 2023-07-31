
from jetson_containers import L4T_VERSION

if L4T_VERSION.major <= 32:
    package['dockerfile'] = 'Dockerfile.jp4'
    package['depends'].extend(['rust', 'protobuf:apt'])
