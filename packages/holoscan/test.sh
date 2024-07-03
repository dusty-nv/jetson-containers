#!/bin/bash
echo 'testing Holoscan...'
export PYTHONPATH=$PYTHONPATH:/opt/nvidia/holoscan/python/lib
python3 /opt/nvidia/holoscan-sdk/examples/hello_world/python/hello_world.py

echo 'Holoscan OK'