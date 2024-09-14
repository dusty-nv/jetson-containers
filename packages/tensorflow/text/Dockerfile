#---
# name: tensorflow_text
# config: config.py
# depends: [tensorflow2]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TENSORFLOW_VERSION \
    CUDAARCHS

RUN git clone --recursive --branch=v${TENSORFLOW_VERSION} --depth=1 https://github.com/tensorflow/text /opt/tensorflow-text

COPY install.sh build.sh /tmp/tensorflow-text/

RUN RUN /tmp/tensorflow-text/install.sh || /tmp/tensorflow-text/build.sh