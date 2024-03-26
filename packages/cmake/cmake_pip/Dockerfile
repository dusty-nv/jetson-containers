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
    && pip3 install --upgrade --force-reinstall --no-cache-dir --verbose cmake \
    \
    && cmake --version \
    && which cmake
