#---
# name: torch_tensorrt
# group: ml
# config: config.py
# depends: [pytorch, torchvision, bazel]
# test: [test.py, test.sh]
# notes: https://pytorch.org/TensorRT/getting_started/installation.html#installation
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCH_TRT_VERSION=main

#
# https://github.com/pytorch/TensorRT/commit/9b0b8f1ccac81050c08091f74870092cbac91ac0
#   pip3 install --no-cache-dir --verbose pyyaml
#   python3 -m pip wheel --verbose --pre . --config-setting="--jetpack-version=5.0" --config-settings="--use-cxx11-abi=1" -w dist && \
#
RUN git clone --branch=${TORCH_TRT_VERSION} --depth=1 https://github.com/pytorch/TensorRT torch_tensorrt && \
    cd torch_tensorrt && \
    cp toolchains/jp_workspaces/WORKSPACE.jp50 WORKSPACE && \
    cd py && \
    sed 's|^        \"torch >=.*|        \"torch\"|g' -i setup.py && \
    sed 's|^    python_requires=.*|    python_requires=\">=3.6\",|g' -i setup.py && \
    cat setup.py && \    
    #python3 setup.py install --jetpack-version 5.0 --use-cxx11-abi && \
    python3 setup.py bdist_wheel --jetpack-version 5.0 --use-cxx11-abi && \
    cp dist/torch_tensorrt*.whl /opt && \
    cd ../../ && \
    rm -rf torch_tensorrt

RUN pip3 install --no-cache-dir --verbose /opt/torch_tensorrt*.whl
    
# ./torchtrtc: error while loading shared libraries: libtorch.so: cannot open shared object file
ARG PYTHON_VERSION
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib
ENV PATH=${PATH}:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch_tensorrt/bin

# make sure it loads
RUN pip3 show torch_tensorrt && python3 -c 'import torch_tensorrt; print(torch_tensorrt.__version__)'