#---
# name: build-essential
# group: build
# notes: installs compilers and build tools
#---
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
	TERM=dumb \
	NUM_CORES=12 \
	CC=/usr/bin/gcc \
	CXX=/usr/bin/g++

RUN set -ex \
	&& apt-get update \
    && apt-get install -y --no-install-recommends \
		build-essential \
		software-properties-common \
		apt-transport-https \
		ca-certificates \
		lsb-release \
		pkg-config \
		gnupg \
		git \
		wget \
		curl \
		nano \
		zip \
		unzip \
	&& apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV MAKEFLAGS=-j$NUM_CORES \
	MAX_JOBS=$NUM_CORES \
	GCC_PATH=$CC \
	CXX_PATH=$CXX \
	CMAKE_CXX_COMPILER=$CXX \
    CMAKE_C_COMPILER=$GCC

RUN gcc --version && \
    g++ --version
