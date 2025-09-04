#---
# name: tensorflow
# group: ml
# depends: [cuda, cudnn, nccl, gdrcopy, nvpl, cusparselt, cudss, nvshmem, numpy, onnx, h5py, bazel, protobuf:cpp, llvm:18]
# test: test.py
# docs: Legacy TF1 installer and TF2 builder
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TENSORFLOW_URL \
    TENSORFLOW_WHL \
    HDF5_DIR="/usr/lib/aarch64-linux-gnu/hdf5/serial/" \
    MAKEFLAGS=-j$(nproc) \
    FORCE_BUILD

COPY install.sh /tmp/tensorflow/

RUN /tmp/tensorflow/install.sh
