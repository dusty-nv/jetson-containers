#!/bin/bash
python3 - <<EOF
print('testing SGLANG...')
from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process
if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

print('SGLANG OK\\n')
EOF
