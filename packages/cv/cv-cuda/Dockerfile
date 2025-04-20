#---
# name: cv-cuda
# alias: cvcuda
# group: cv
# config: config.py
# depends: [pytorch, torchvision, pyav, pycuda, cmake]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NVCV_VERSION \
    NVCV_BINARIES \
    NVCV_PYTHON=on \
    TMP_DIR=/tmp/cv-cuda

COPY install.sh ${TMP_DIR}/
RUN cd ${TMP_DIR} && bash install.sh