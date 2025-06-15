#!/usr/bin/env bash
set -ex

cd /opt
git clone --depth=1 https://github.com/NVIDIA-AI-IOT/torch2trt

cd torch2trt
ls -R /tmp/torch2trt
cp /tmp/torch2trt/flattener.py torch2trt

pip3 install .

sed 's|^set(CUDA_ARCHITECTURES.*|#|g' -i CMakeLists.txt
sed 's|Catch2_FOUND|False|g' -i CMakeLists.txt

cmake -B build \
  -DCUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .

cmake --build build --target install

ldconfig

pip3 install nvidia-pyindex
pip3 install onnx-graphsurgeon
