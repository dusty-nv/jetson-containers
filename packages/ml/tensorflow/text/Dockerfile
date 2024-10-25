#---
# name: tensorflow_text
# config: config.py
# depends: [tensorflow2]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TENSORFLOW_TEXT_VERSION \
    PYTHON_VERSION_MAJOR \
    PYTHON_VERSION_MINOR \
    CUDAARCHS

COPY install.sh build.sh /tmp/tensorflow-text/

RUN /tmp/tensorflow-text/install.sh || /tmp/tensorflow-text/build.sh