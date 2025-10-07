#---
# name: zed
# group: hardware
# config: config.py
# depends: [cuda, cudastack:standard, python, opengl]
# test: test.py
# notes: Stereolabs ZED SDK
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ZED_URL \
    L4T_MAJOR_VERSION \
    L4T_MINOR_VERSION \
    L4T_PATCH_VERSION

# Enable streaming features on Jetson inside a container
ENV LOGNAME root
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/tegra/libv4l2.so.0 /usr/lib/$(uname -m)-linux-gnu/libv4l2.so

# ZED SDK installer
#echo "# R${L4T_MAJOR_VERSION} (release), REVISION: ${L4T_MINOR_VERSION}.${L4T_PATCH_VERSION}" > /etc/nv_tegra_release ; \
RUN apt-get update -y || true ; apt-get install --no-install-recommends lsb-release wget less zstd udev sudo apt-transport-https -y && \
    wget $WGET_FLAGS -O ZED_SDK_Linux.run ${ZED_URL} && \
    chmod +x ZED_SDK_Linux.run ; ./ZED_SDK_Linux.run silent skip_drivers && \
    rm -rf /usr/local/zed/resources/* \
    rm -rf ZED_SDK_Linux.run && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# ZED Python API
# apt-get update -y || true ; apt-get install --no-install-recommends python3 python3-pip python3-dev python3-setuptools build-essential -y && \
# python3 -m pip install cython wheel numpy
RUN wget $WGET_FLAGS download.stereolabs.com/zedsdk/pyzed -O /usr/local/zed/get_python_api.py && \
    python3 /usr/local/zed/get_python_api.py && \
    python3 -m pip install pyopengl *.whl && \
    rm *.whl && \
    rm -rf /usr/local/zed/resources

# auto-mount ZED model cache
RUN mkdir -p /data/zed/resources && \
    mkdir -p /root/.cache/zed/resources && \
    ln -s /data/zed /root/.cache/zed && \
    ln -s /root/.cache/zed/resources /usr/local/zed/resources
