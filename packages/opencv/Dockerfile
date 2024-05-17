#---
# name: opencv
# group: core
# config: config.py
# depends: [cuda, cudnn, python, numpy]
# test: test.py
# notes: install or build OpenCV (with CUDA) from Jetson pip server
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENCV_VERSION \
    OPENCV_PYTHON \
    OPENCV_URL \
    CUDA_ARCH_BIN \
    FORCE_BUILD=off

ENV OPENCV_VERSION=${OPENCV_VERSION} \
    OPENCV_URL=${OPENCV_URL}
    
COPY build.sh \
     install.sh \
     install_deps.sh \
     install_deb.sh \
     install_pip.sh \
     patches.diff \
     /tmp/opencv/

RUN cd /tmp/opencv && ./install.sh || ./build.sh

