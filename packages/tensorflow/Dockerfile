#---
# name: tensorflow
# group: ml
# depends: [cuda, cudnn, tensorrt, python, numpy, protobuf:cpp]
# test: test.py
# docs: |
#   Container for TF1/TF2 with CUDA support.
#   Note that the [`l4t-tensorflow`](/packages/l4t/l4t-tensorflow) containers are similar, with the addition of OpenCV and PyCUDA.  
#
#   The TensorFlow wheels used in these are from https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform
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