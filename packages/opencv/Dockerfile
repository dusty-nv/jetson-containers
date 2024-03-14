#---
# name: opencv
# group: core
# config: config.py
# depends: [cuda, cudnn, python, numpy]
# test: test.py
# notes: install OpenCV (with CUDA) from binaries built by `opencv_builder`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENCV_URL \
    OPENCV_DEB

COPY opencv_install.sh /opt/opencv_install.sh

# leave the installer script under /opt and patch it with the
# build args hardcoded in case it needs re-installed later
RUN set -ex \
    && sed -i \
        -e '2i OPENCV_URL=${OPENCV_URL}' \
        -e '3i OPENCV_DEB=${OPENCV_DEB}' \
        /opt/opencv_install.sh \
    && /opt/opencv_install.sh
