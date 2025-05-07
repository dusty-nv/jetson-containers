#---
# name: open3d
# alias: open3d
# group: cv
# config: config.py
# depends: [pytorch, torchvision, torchaudio, pyav, pycuda, cmake, ninja, llvm:20]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPEN3D_VERSION \
    TMP_DIR=/tmp/open3d

COPY build.sh install.sh "${TMP_DIR}"/
RUN "${TMP_DIR}"/install.sh || "${TMP_DIR}"/build.sh || touch "${TMP_DIR}"/.build.failed

ENV LD_PRELOAD=/usr/local/lib/libOpen3D.so.0.19