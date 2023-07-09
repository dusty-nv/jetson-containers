#
# name: opencv
# config: config.py
# depends: numpy
# test: test.py
# description: install OpenCV (with CUDA) from binaries built by opencv_builder
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENCV_URL
ARG OPENCV_DEB

COPY opencv_install.sh /tmp/opencv_install.sh
RUN cd /tmp && ./opencv_install.sh ${OPENCV_URL} ${OPENCV_DEB}
