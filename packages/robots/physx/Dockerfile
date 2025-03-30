#---
# name: physx
# group: robots
# depends: [cuda, python, cmake]
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
    && pip3 install mujoco
    
RUN git clone --recursive --depth=1 https://github.com/NVIDIA-Omniverse/PhysX /opt/PhysX && \
    cd /opt/PhysX/physx && \
    bash generate_projects.sh linux-aarch64 && \
    cd compiler/linux-aarch64-release && \
    make -j$(nproc) && \
    make install
