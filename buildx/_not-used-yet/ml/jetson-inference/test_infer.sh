#!/usr/bin/env bash
set -e

echo "testing jetson-inference..."

python3 -c 'import jetson_inference'
python3 -c 'import jetson_utils'

detectnet --help
detectnet.py --help

echo "jetson-inference OK"
