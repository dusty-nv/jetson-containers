
from jetson_containers import L4T_VERSION

if L4T_VERSION.major >= 34:
    package['test'].append('test_gpt.py')
