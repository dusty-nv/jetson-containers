
from jetson_containers import L4T_VERSION

if L4T_VERSION.major >= 36:
    package['depends'].append('xtts')
