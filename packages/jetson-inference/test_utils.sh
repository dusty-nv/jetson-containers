#!/usr/bin/env bash
set -e

echo "testing jetson-utils..."

cd /opt/jetson-inference/build/aarch64/bin

mkdir -p images/test

wget https://raw.githubusercontent.com/dusty-nv/jetson-inference/master/data/images/granny_smith_1.jpg -O /tmp/apple.jpg

python3 cuda-examples.py /tmp/apple.jpg /tmp/test.jpg

python3 cuda-from-numpy.py --filename=/tmp/test.jpg

python3 cuda-to-numpy.py

echo "jetson-utils OK"
