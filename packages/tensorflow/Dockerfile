#---
# name: tensorflow
# group: ml
# depends: [python, numpy, protobuf:cpp]
# test: test.py
# docs: |
#   Container for TF1/TF2 with CUDA support.
#   Note that the [`l4t-tensorflow`](/packages/l4t/l4t-tensorflow) containers are similar, with the addition of OpenCV and PyCUDA.  
#
#   The TensorFlow wheels used in these are from https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


# setup environment
ENV DEBIAN_FRONTEND=noninteractive
ARG HDF5_DIR="/usr/lib/aarch64-linux-gnu/hdf5/serial/"
ARG MAKEFLAGS=-j$(nproc)

RUN printenv


# install prerequisites - https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#prereqs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  liblapack-dev \ 
		  libblas-dev \
		  libhdf5-serial-dev \
		  hdf5-tools \
		  libhdf5-dev \
		  zlib1g-dev \
		  libjpeg8-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir setuptools Cython wheel


# install Python TF dependencies
RUN H5PY_SETUP_REQUIRES=0 pip3 install --no-cache-dir --verbose h5py
RUN pip3 install --no-cache-dir --verbose future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 futures pybind11


# TensorFlow 
ARG TENSORFLOW_URL
ARG TENSORFLOW_WHL

RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${TENSORFLOW_URL} -O ${TENSORFLOW_WHL} && \
    pip3 install --no-cache-dir --verbose ${TENSORFLOW_WHL} && \
    rm ${TENSORFLOW_WHL}
