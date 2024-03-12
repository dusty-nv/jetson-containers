#---
# name: build-essential
# group: build
# notes: installs compilers and build tools
#---
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
	LANGUAGE=en_US:en \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
	TERM=dumb \
	NUM_CORES=12 \
	CC=/usr/bin/gcc \
	CXX=/usr/bin/g++

RUN set -ex \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends \
		locales \
		locales-all \
		tzdata \
	&& locale-gen en_US $LANG \
	&& update-locale LC_ALL=$LC_ALL LANG=$LANG \
	&& locale \
    && apt-get install -y --no-install-recommends \
		build-essential \
		software-properties-common \
		apt-transport-https \
		ca-certificates \
		lsb-release \
		pkg-config \
		gnupg \
		git \
		gdb \
		wget \
		curl \
		nano \
		zip \
		unzip \
		time \
	&& apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
	&& gcc --version \
	&& g++ --version


ENV MAKEFLAGS=-j$NUM_CORES \
	MAX_JOBS=$NUM_CORES \
	GCC_PATH=$CC \
	CXX_PATH=$CXX \
	CMAKE_CXX_COMPILER=$CXX \
    CMAKE_C_COMPILER=$GCC
