#---
# name: physx
# group: sim
# depends: [cuda, python, cmake]
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_ARCHS="87;90;100;103;110;120;121"

# https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/documentation/platformreadme/linux/README_LINUX.md
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  freeglut*-dev \
		  libglu1-mesa-dev \
		  libxdamage-dev \
		  libxmuu-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && uv pip install mujoco

COPY install.sh /tmp/physx/install.sh

RUN /tmp/physx/install.sh
