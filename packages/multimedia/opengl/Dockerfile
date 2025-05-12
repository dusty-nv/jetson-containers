#---
# name: opengl
# group: multimedia
# depends: [build-essential]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglu1-mesa-dev \
        libglx-mesa0 \
        libegl-dev \
        libxrender1 \
        libglfw3-dev \
        libglew-dev \
        glew-utils \
        mesa-common-dev \
        freeglut3-dev  \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
