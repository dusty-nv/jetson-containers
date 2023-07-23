#---
# name: opencv
# alias: opencv_cuda
# group: core
# config: config.py
# depends: [python, numpy]
# test: test.py
# notes: install OpenCV (with CUDA) from binaries built by opencv_builder
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENCV_URL
ARG OPENCV_DEB

COPY opencv_install.sh /opt/opencv_install.sh

# leave the installer script under /opt and patch it with the
# build args hardcoded in case it needs re-installed later
RUN cd /opt && \
    sed -i "2i OPENCV_URL=${OPENCV_URL}" opencv_install.sh && \
    sed -i "3i OPENCV_DEB=${OPENCV_DEB}" opencv_install.sh && \
    cat opencv_install.sh

RUN cd /opt && ./opencv_install.sh
