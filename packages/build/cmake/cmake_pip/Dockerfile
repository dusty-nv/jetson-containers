#---
# name: cmake:pip
# alias: cmake
# group: build
# depends: [build-essential, python]
# notes: upgrade `cmake` with `pip`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN set -ex \
    && pip3 install --force-reinstall cmake \
    && cmake --version \
    && which cmake
