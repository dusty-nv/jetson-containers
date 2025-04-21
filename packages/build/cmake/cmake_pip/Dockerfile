#---
# name: cmake:pip
# alias: cmake
# group: build
# depends: [build-essential, python]
# notes: upgrade `cmake` with `pip`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

#ENV CMAKE_POLICY_VERSION_MINIMUM="3.10"
COPY install.sh /tmp/cmake/install.sh
RUN /tmp/cmake/install.sh
