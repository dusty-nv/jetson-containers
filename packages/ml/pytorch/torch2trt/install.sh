#!/usr/bin/env bash
set -ex

cd /opt
git clone --depth=1 https://github.com/NVIDIA-AI-IOT/torch2trt

cd torch2trt
ls -R /tmp/torch2trt
cp /tmp/torch2trt/flattener.py torch2trt


# Install TensorRT Wheel First to ensure libs are present
TRT_WHEEL=$(find /usr -name "tensorrt-*-cp310-*-linux_aarch64.whl" -print -quit)

if [ -f "$TRT_WHEEL" ]; then
    echo "Installing existing TensorRT wheel: $TRT_WHEEL"
    uv pip install "$TRT_WHEEL"
else
    echo "CRITICAL: TensorRT wheel not found. Build cannot proceed."
    exit 1
fi

python3 setup.py install --plugins

sed 's|^set(CUDA_ARCHITECTURES.*|#|g' -i CMakeLists.txt
sed 's|Catch2_FOUND|False|g' -i CMakeLists.txt

cmake -B build \
  -DCUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .

cmake --build build --target install

ldconfig

uv pip install --no-build-isolation onnx-graphsurgeon
