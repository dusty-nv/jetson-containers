#---
# name: tensorflow
# group: ml
# depends: [cuda, cudnn, tensorrt, python, pybind11, numpy, h5py, bazel, protobuf:cpp]
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