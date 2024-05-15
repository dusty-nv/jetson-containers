#---
# name: eudev
# group: smart-home
# config: config.py
# test: test.sh
# requires: '>=34.1.0'
# depends: [build-essential]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG EUDEV_VERSION

WORKDIR /

COPY *.sh /tmp/eudev/

RUN /tmp/eudev/build.sh
