#---
# name: pyceres
# group: nerf
# config: config.py
# depends: [python, ninja, cmake, numpy]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/cvg/pyceres.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYCERES_VERSION \
FORCE_BUILD=off

ARG CUDAARCHS

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    libgtest-dev \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*


COPY build.sh install.sh /tmp/PYCERES/

RUN cd /opt/
RUN pip3 install build wheel

RUN git clone --branch 2.2.0 https://ceres-solver.googlesource.com/ceres-solver.git --single-branch /opt/ceres-solver && \
    cd /opt/ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cd ../.. && \
    rm -rf ceres-solver

RUN /tmp/PYCERES/install.sh || /tmp/PYCERES/build.sh
