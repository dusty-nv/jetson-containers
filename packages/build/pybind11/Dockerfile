#---
# name: pybind11
# alias: pybind11:global
# group: build
# depends: [python]
# test: test.sh
# notes: https://github.com/pybind/pybind11
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY install.sh /tmp/pybind11/
RUN /tmp/pybind11/install.sh
