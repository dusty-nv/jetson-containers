#---
# name: opencv
# group: cv
# config: config.py
# depends: [cuda, cudastack:standard, python, numpy, opengl, vulkan, video-codec-sdk, ffmpeg]
# test: test.py
# notes: https://github.com/opencv/opencv-python
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENCV_VERSION \
    OPENCV_PYTHON \
    OPENCV_URL \
    CUDA_ARCH_BIN \
    FORCE_BUILD=off \
    TMP=/tmp/opencv

ENV OPENCV_VERSION=${OPENCV_VERSION} \
    OPENCV_URL=${OPENCV_URL}

COPY build.sh \
     install.sh \
     install_deps.sh \
     install_deb.sh \
     patches.diff \
     $TMP/

RUN cd $TMP && ./install.sh || ./build.sh || true

RUN if [ ! -f "$TMP/.opencv" ]; then \
      echo "FAILED to install OpenCV $OPENCV_VERSION"; \
      exit 1; \
    fi
