#
# jetson-containers build genesis:builder
# OPTIONAL_PERMISSION_ARGS=true jetson-containers run $(autotag genesis:builder)
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GENESIS_VERSION \
    FORCE_BUILD=on

RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    libx11-6 \
    libgl1-mesa-glx \
    libxrender1 \
    libglu1-mesa \
    libglib2.0-0 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the build and install scripts to the container
COPY build.sh install.sh /tmp/GENESIS/

CMD ["/bin/bash", "-c", "/tmp/GENESIS/install.sh || /tmp/GENESIS/build.sh"]
