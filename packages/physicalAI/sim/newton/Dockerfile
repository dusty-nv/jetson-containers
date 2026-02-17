#---
# name: newton
# group: sim
# depends: [cuda, python, cmake, warp]
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

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

RUN git clone --recursive --depth=1 https://github.com/newton-physics/newton /opt/newton && \
    cd /opt/newton && \
    uv sync --extra examples
