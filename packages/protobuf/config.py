
from jetson_containers import L4T_VERSION

if L4T_VERSION.major >= 35:   # JetPack 5.0.2 / 5.1.x
    PROTOBUF_VERSION = '3.20.3'
elif L4T_VERSION.major == 34:  # JetPack 5.0 / 5.0.1
    PROTOBUF_VERSION = '3.20.1'
elif L4T_VERSION.major == 32:  # JetPack 4
    PROTOBUF_VERSION = '3.19.4'

package['build_args'] = {
    'PROTOBUF_VERSION': PROTOBUF_VERSION,
}
