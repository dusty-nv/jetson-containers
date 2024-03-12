#---
# name: jetson-inference
# group: ml
# depends: [pytorch, torchvision, cmake, gstreamer]
# test: [test_utils.sh, test_infer.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG JETSON_INFERENCE_REPO=dusty-nv/jetson-inference
ARG JETSON_INFERENCE_SOURCE=/opt/jetson-inference

ADD https://api.github.com/repos/${JETSON_INFERENCE_REPO}/git/refs/heads/master /tmp/jetson_inference_version.json

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		    libglew-dev \
		    glew-utils \
		    libsoup2.4-dev \
		    libjson-glib-dev \
		    libgstrtspserver-1.0-dev \
		    avahi-utils && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    \
    git clone --recursive --depth=1 https://github.com/${JETSON_INFERENCE_REPO} ${JETSON_INFERENCE_SOURCE} && \
    cd ${JETSON_INFERENCE_SOURCE} && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j$(nproc) && \
    make install && \
    /bin/bash -O extglob -c "cd /jetson-inference/build; rm -rf -v !($(uname -m)|download-models.*)" && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    \
    cd ${JETSON_INFERENCE_SOURCE}/examples/my-recognition && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make && \
    \
    cd ${JETSON_INFERENCE_SOURCE} && \
    pip3 install --no-cache-dir --ignore-installed blinker && \
    pip3 install --no-cache-dir --verbose -r python/training/detection/ssd/requirements.txt && \
    pip3 install --no-cache-dir --verbose -r python/www/flask/requirements.txt  && \
    pip3 install --no-cache-dir --verbose -r python/www/dash/requirements.txt && \
    \
    ln -s ${JETSON_INFERENCE_SOURCE} /jetson-inference
    
RUN python3 -c 'import jetson_inference; import jetson_utils'
