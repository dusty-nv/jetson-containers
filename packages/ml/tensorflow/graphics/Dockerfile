#---
# name: tensorflow_graphics
# group: tensorflow
# config: config.py
# depends: [tensorflow2]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/tensorflow/graphics
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TENSORFLOW_GRAPHICS_VERSION \
    PYTHON_VERSION_MAJOR \
    PYTHON_VERSION_MINOR \
    CUDAARCHS

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenexr-dev
    
COPY build.sh install.sh /tmp/GRAPHICS/

RUN /tmp/GRAPHICS/install.sh || /tmp/GRAPHICS/build.sh
