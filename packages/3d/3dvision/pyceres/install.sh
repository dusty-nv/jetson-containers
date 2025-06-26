#!/usr/bin/env bash
set -ex

# apt dependencies for NERF/3DGS stack
apt-get update
apt-get install -y --no-install-recommends \
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
	libgoogle-glog-dev \
	libmetis-dev \
	libprotobuf-dev \
	libqt5opengl5-dev \
	libsqlite3-dev \
	libsuitesparse-dev \
	protobuf-compiler \
	libgtest-dev \
	qtbase5-dev \
	sudo \
	vim-tiny
	
rm -rf /var/lib/apt/lists/*
apt-get clean

# build the ceres-solver C++ library
git clone --branch 2.2.0 https://ceres-solver.googlesource.com/ceres-solver.git --single-branch /opt/ceres-solver && \
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

# either install or build pyceres wheel
if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of pyceres ${PYCERES}"
	exit 1
fi

pip3 install pyceres==${PYCERES_VERSION}