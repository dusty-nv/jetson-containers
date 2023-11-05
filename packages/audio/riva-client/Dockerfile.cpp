#---
# name: 'riva-client:cpp'
# group: audio
# depends: [bazel]
# requires: '>=34.1.0'
# test: [test.sh]
# docs: docs.md
# notes: https://github.com/nvidia-riva/cpp-clients
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


WORKDIR /opt/riva


# install prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libasound2-dev \
		  libopus0 \
		  libopus-dev \
		  libopusfile0 \
		  libopusfile-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
    
# Riva C++ client (binaries in bazel-bin/riva/clients)
ADD https://api.github.com/repos/nvidia-riva/cpp-clients/git/refs/heads/main /tmp/riva_cpp_version.json

RUN git clone --depth=1 --recursive https://github.com/nvidia-riva/cpp-clients && \
    cd cpp-clients && \
    bazel build ...
    
WORKDIR /