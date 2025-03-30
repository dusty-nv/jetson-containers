#---
# name: protobuf:cpp
# alias: protobuf
# group: build
# config: config.py
# depends: [build-essential, python]
# test: [test.py, test.sh]
# notes: build protobuf using cpp implementation (https://jkjung-avt.github.io/tf-trt-revisited/)
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PROTOBUF_VERSION=3.19.4
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp

COPY install.sh /tmp/protobuf/
RUN /tmp/protobuf/install.sh
