#---
# name: pymeshlab
# group: nerf
# config: config.py
# depends: [python, cmake, ninja, numpy, polyscope]
# requires: '>=34.1.0'
# notes: https://github.com/cnr-isti-vclab/PyMeshLab.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYMESHLAB_VERSION \
    FORCE_BUILD=off

RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    fontconfig \
    fuse \
    gdb \
    git \
    kmod \
    libboost-dev \
    libdbus-1-3 \
    libegl-dev \
    libfuse2 \
    libgmp-dev \
    libglu1-mesa-dev \
    libmpfr-dev \
    libpulse-mainloop-glib0 \
    libtbb-dev \
    libxcb-icccm4-dev \
    libxcb-image0-dev \
    libxcb-keysyms1-dev \
    libxcb-render-util0-dev \
    libxcb-shape0 \
    libxcb-xinerama0-dev \
    libxcb-xkb-dev \
    libxkbcommon-x11-dev \
    libxerces-c-dev \
    patchelf \
    rsync \
    libeigen3-dev \
    qtbase5-dev \
    qtbase5-dev-tools \
    libqt5opengl5-dev \
    mesa-common-dev \
    make \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the build and install scripts to the container
COPY build.sh install.sh /tmp/PYMESHLAB/

# Prepare the extra files to be copied later
RUN mkdir -p /tmp/PYMESHLAB/extra
COPY extra/ /tmp/PYMESHLAB/extra
RUN ls -la /tmp/PYMESHLAB/extra*

# Clone the PyMeshLab repository
RUN git clone --branch=v${PYMESHLAB_VERSION}  --depth=1 --recursive https://github.com/cnr-isti-vclab/PyMeshLab /opt/pymeshlab

# Navigate to the /opt/pymeshlab/src/meshlab/resources/linux directory
WORKDIR /opt/pymeshlab/src/meshlab/resources/linux

# Remove all content in the linux folder
RUN rm -rf linuxdeploy linuxdeploy-plugin-appimage linuxdeploy-plugin-qt

# Copy the necessary files from /tmp/PYMESHLAB/extra
RUN cp -r /tmp/PYMESHLAB/extra/* /opt/pymeshlab/src/meshlab/resources/linux

# Set execute permissions on the specified files
RUN chmod +x linuxdeploy linuxdeploy-plugin-appimage linuxdeploy-plugin-qt

# Set the command to run the install or build script when the container starts
CMD ["/bin/bash", "-c", "/tmp/PYMESHLAB/install.sh || /tmp/PYMESHLAB/build.sh"]
