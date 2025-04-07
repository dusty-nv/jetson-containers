#!/usr/bin/env bash
set -e

echo "testing jetson-utils..."

if [ -d "/usr/local/cuda" ]; then
  cd /opt/jetson-utils/python/examples

  mkdir -p images/test

  wget https://raw.githubusercontent.com/dusty-nv/jetson-inference/master/data/images/granny_smith_1.jpg -O /tmp/apple.jpg

  python3 cuda-examples.py /tmp/apple.jpg /tmp/test.jpg

  python3 cuda-from-numpy.py --filename=/tmp/test.jpg

  python3 cuda-to-numpy.py
fi

if [ -d "/opt/jetson-utils/python/jetson_utils" ]; then
  python3 -m jetson_utils.test
fi

echo "jetson-utils OK"