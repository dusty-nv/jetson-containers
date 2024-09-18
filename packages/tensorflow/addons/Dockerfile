#---
# name: tensorflow_addons
# group: tensorflow
# config: config.py
# depends: [tensorflow2]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/tensorflow/addons
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TENSORFLOW_ADDONS_VERSION \
    CUDAARCHS
    
COPY build.sh install.sh /tmp/addons/

RUN /tmp/addons/install.sh || /tmp/addons/build.sh
