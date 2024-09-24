#---
# name: tensorflow_text
# config: config.py
# depends: [bazel, tensorflow2]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TENSORFLOW_VERSION \
    CUDAARCHS

COPY install.sh build.sh /tmp/tensorflow-text/

RUN /tmp/tensorflow-text/install.sh || /tmp/tensorflow-text/build.sh