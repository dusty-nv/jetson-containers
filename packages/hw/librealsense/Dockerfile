#---
# name: librealsense
# group: hardware
# depends: [cuda, python, cmake]
# test: test.py
# notes: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FORCE_BUILD=off \
    LIBREALSENSE_VERSION \
    LIBREALSENSE_VERSION_SPEC


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl-dev \
		libusb-1.0-0-dev \
        libgtk-3-dev \
        libglfw3-dev \
		libgl1-mesa-dev \
		libglu1-mesa-dev \
		qtcreator \
		udev \
        git \
        python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# https://github.com/IntelRealSense/librealsense/issues/11931
# ARG LIBREALSENSE_VERSION=development

# https://github.com/dusty-nv/jetson-containers/issues/398

COPY build.sh install.sh /tmp/librealsense/

RUN /tmp/librealsense/install.sh || /tmp/librealsense/build.sh
