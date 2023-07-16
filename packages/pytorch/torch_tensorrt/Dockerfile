#
# name: torch_tensorrt
# category: ml
# config: config.py
# depends: [pytorch, torchvision, bazel]
# test: [test.py, test.sh]
# notes: https://pytorch.org/TensorRT/getting_started/installation.html#installation
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN git clone --depth=1 https://github.com/pytorch/TensorRT torch_tensorrt && \
    cd torch_tensorrt && \
    cp toolchains/jp_workspaces/WORKSPACE.jp50 WORKSPACE && \
    cd py && \
    python3 setup.py install --jetpack-version 5.0 --use-cxx11-abi && \
    cd ../../ && \
    rm -rf torch_tensorrt
    
# ./torchtrtc: error while loading shared libraries: libtorch.so: cannot open shared object file
ARG PYTHON_VERSION
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib
ENV PATH=${PATH}:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch_tensorrt/bin