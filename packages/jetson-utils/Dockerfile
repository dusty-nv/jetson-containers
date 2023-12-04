#---
# name: jetson-utils
# group: cuda
# depends: [cuda, tensorrt, python, cmake, gstreamer]
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ARG JETSON_UTILS_REPO=dusty-nv/jetson-utils
ARG JETSON_UTILS_BRANCH=master

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		    libglew-dev \
		    glew-utils \
		    libsoup2.4-dev \
		    libjson-glib-dev \
		    libgstrtspserver-1.0-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
ADD https://api.github.com/repos/${JETSON_UTILS_REPO}/git/refs/heads/${JETSON_UTILS_BRANCH} /tmp/jetson_utils_version.json

RUN git clone --branch=${JETSON_UTILS_BRANCH} --depth=1 https://github.com/${JETSON_UTILS_REPO}

RUN cd jetson-utils && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j$(nproc) && \
    make install
    
RUN python3 -c 'import jetson_utils'
