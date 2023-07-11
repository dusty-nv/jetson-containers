
from jetson_containers import L4T_VERSION

if L4T_VERSION.major < 34:   # JetPack 5
    print("-- bitsandbytes isn't available for JetPack 4, disabling...")
    package = None
