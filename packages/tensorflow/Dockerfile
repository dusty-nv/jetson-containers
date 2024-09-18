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
    TENSORFLOW_VERSION \
    HDF5_DIR="/usr/lib/aarch64-linux-gnu/hdf5/serial/" \
    MAKEFLAGS=-j$(nproc) \
    FORCE_BUILD=off

ENV PYTHON_VERSION=${PYTHON_VERSION} \
    TENSORFLOW_URL=${TENSORFLOW_URL} \
    TENSORFLOW_WHL=${TENSORFLOW_WHL} \
    TENSORFLOW_VERSION=${TENSORFLOW_VERSION} \
    HDF5_DIR=${HDF5_DIR} \
    MAKEFLAGS=${MAKEFLAGS} \
    FORCE_BUILD=${FORCE_BUILD}

# copy installation and build scripts for tensorflow
COPY build.sh install.sh link_cuda.sh /tmp/tensorflow/

# attempt to install tensorflow from pip, and fall back to building it if the installation fails
RUN /tmp/tensorflow/install.sh || /tmp/tensorflow/build.sh
