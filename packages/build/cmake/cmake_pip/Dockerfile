#---
# name: cmake:pip
# alias: cmake
# group: build
# depends: [build-essential, python]
# notes: upgrade `cmake` with `pip`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY install.sh /tmp/cmake/install.sh
RUN /tmp/cmake/install.sh
