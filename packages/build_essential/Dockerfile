#
# name: build_essential
# category: base
# description: installs compilers and build tools
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  build-essential \
		  git \
		  wget \
		  curl \
		  nano \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
RUN gcc --version && \
    g++ --version
