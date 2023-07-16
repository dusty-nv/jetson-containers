#
# name: torch_tensorrt
# category: ml
# depends: [pytorch, bazel]
# test: test.py
# notes: https://pytorch.org/TensorRT/getting_started/installation.html#installation
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

#COPY WORKSPACE /tmp/WORKSPACE

RUN git clone --depth=1 https://github.com/pytorch/TensorRT torch_tensorrt && \
    cd torch_tensorrt && \
    cp toolchains/jp_workspaces/WORKSPACE.jp50 WORKSPACE && \
    cd py && \
    python3 setup.py install --jetpack-version 5.0 --use-cxx11-abi && \
    cd ../../ && \
    rm -rf torch_tensorrt
    