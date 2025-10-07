#---
# name: nerfstudio
# group: nerf
# config: config.py
# depends: [transformers, bitsandbytes, diffusers, h5py, ninja, ffmpeg, polyscope, pymeshlab, tinycudann, nerfacc, pycolmap, glomap, hloc, pyceres, gsplat, nerfview]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/nerfstudio-project/nerfstudio
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NERFSTUDIO_VERSION \
FORCE_BUILD=off

ARG CUDAARCHS

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libboost-filesystem1.74.0 \
    libboost-program-options1.74.0 \
    libc6 \
    libceres-dev \
    libfreeimage3 \
    libgcc-s1 \
    libgl1 \
    libglew2.2 \
    libgoogle-glog-dev \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    build-essential \
    python-is-python3 \
    ffmpeg \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN cd /opt/
RUN uv pip install build wheel
RUN export MAKEFLAGS="-j$(nproc)"

# Add glog path to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

RUN python3 -m pip install --upgrade pip setuptools promise omegaconf

EXPOSE 7007

COPY build.sh install.sh /tmp/NERFSTUDIO/
RUN /tmp/NERFSTUDIO/install.sh || /tmp/NERFSTUDIO/build.sh

RUN cd /opt/
CMD ["/bin/bash"]
