#---
# name: meshlab
# group: nerf
# config: config.py
# depends: [python, cmake, ninja, numpy]
# requires: '>=34.1.0'
# notes: https://github.com/cnr-isti-vclab/MeshLab.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MESHLAB_VERSION \
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
file \
desktop-file-utils \
&& rm -rf /var/lib/apt/lists/*

# Download and set up appimagetool for aarch64
RUN wget https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-aarch64.AppImage \
    && chmod +x appimagetool-aarch64.AppImage \
    && mv appimagetool-aarch64.AppImage /usr/local/bin/appimagetool


COPY build.sh /tmp/MESHLAB/

RUN mkdir -p /tmp/MESHLAB/extra
COPY extra/ /tmp/MESHLAB/extra
RUN ls -la /tmp/MESHLAB/extra*

RUN git clone --branch=${MESHLAB_VERSION} --depth=1 --recursive https://github.com/cnr-isti-vclab/meshlab /opt/meshlab || \
    git clone --depth=1 --recursive https://github.com/cnr-isti-vclab/meshlab /opt/meshlab


# Navigate to the /opt/meshlab/src/meshlab/resources/linux directory
WORKDIR /opt/meshlab/resources/linux

# Remove all content in the linux folder
RUN rm -rf linuxdeploy linuxdeploy-plugin-appimage linuxdeploy-plugin-qt

# Copy the necessary files from /tmp/MESHLAB/extra
RUN cp -r /tmp/MESHLAB/extra/* /opt/meshlab/resources/linux

# Set execute permissions on the specified files
RUN chmod +x linuxdeploy linuxdeploy-plugin-appimage linuxdeploy-plugin-qt

# Set the command to run the install or build script when the container starts
CMD ["/bin/bash", "-c", "/tmp/MESHLAB/build.sh"]

RUN cd /opt/