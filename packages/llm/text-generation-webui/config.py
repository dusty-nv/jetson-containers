
from jetson_containers import L4T_VERSION

if L4T_VERSION.major <= 35:
    package['depends'].append('bitsandbytes')
    package['build_args'] = {
        'LD_PRELOAD_LIBS': '/usr/local/lib/python3.8/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0'
    }