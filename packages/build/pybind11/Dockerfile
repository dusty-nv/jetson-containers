#---
# name: pybind11
# alias: pybind11:global
# group: build
# depends: [python]
# test: test.sh
# notes: installs more recent pybind11 from pip
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY install.sh /tmp/pybind11/
RUN /tmp/pybind11/install.sh
