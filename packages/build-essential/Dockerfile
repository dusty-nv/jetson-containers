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
    LC_ALL=en_US.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends locales locales-all tzdata && \
    locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    locale && \
    apt-get install -y --no-install-recommends \
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
		  time && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    gcc --version && \
    g++ --version
