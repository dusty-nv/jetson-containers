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

COPY build.sh \
     install_deps.sh \
	install_deb.sh \
	install_pip.sh \
     patches.diff \
	/tmp/opencv/

RUN cd /tmp/opencv && \
    ./install_deps.sh && \
    ./install_pip.sh || \
    ./install_deb.sh || \
    ./build.sh

# from when opencv was installed via debian tarball:
# leave the installer script under /opt and patch it with the
# build args hardcoded in case it needs re-installed later
#RUN set -ex \
#    && sed -i \
#        -e '2i OPENCV_URL=${OPENCV_URL}' \
#        -e '3i OPENCV_DEB=${OPENCV_DEB}' \
#        /opt/opencv_install.sh \
#    && /opt/opencv_install.sh
