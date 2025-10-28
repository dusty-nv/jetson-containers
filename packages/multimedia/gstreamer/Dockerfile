#---
# name: gstreamer
# group: multimedia
# depends: opencv
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# From https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html#gstreamer-1-0-installation-and-set-up
RUN apt-get update \
    && apt-get install -y gstreamer1.0-tools gstreamer1.0-alsa \
     gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
     gstreamer1.0-libav \
    && apt-get install -y libgstreamer1.0-dev \
     libgstreamer-plugins-base1.0-dev \
     libgstreamer-plugins-good1.0-dev \
     libgstreamer-plugins-bad1.0-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -s /usr/lib/python3/dist-packages/gi /opt/venv/lib/python3.*/site-packages/ # make gi available in the venv
