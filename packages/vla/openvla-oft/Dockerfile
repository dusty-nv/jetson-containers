#---
# name: openvla-oft
# group: vla
# config: config.py
# depends: [transformers, flash-attention, bitsandbytes, tensorboard, h5py]
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install prerequisites then copy and install NCCL local-repo .deb (offline/local).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      gnupg \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the nccl local-repo deb (ensure you placed it in the repo as instructed)
COPY --chown=root:root ../../cuda/nccl/nccl-local-repo-*.deb /tmp/nccl/
RUN dpkg -i /tmp/nccl/nccl-local-repo-*.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends libnccl2 libnccl-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/nccl && \
    true

ADD https://api.github.com/repos/explicitcontextualunderstanding/openvla/git/refs/heads/main /tmp/openvla_version.json

RUN git clone --depth=1 --recursive https://github.com/explicitcontextualunderstanding/openvla /opt/openvla && \
    cd /opt/openvla && \
    pip3 install -e . && \
    pip3 install --no-deps git+https://github.com/moojink/openvla-oft

# patch issue of NCCL P2P not being supported on Jetson, and accelerate not honoring alternate backends
RUN PYTHON_ROOT=`pip3 show accelerate | grep Location: | cut -d' ' -f2` && \
    ACCELERATE_STATE="$PYTHON_ROOT/accelerate/state.py" && \
    echo "patching $ACCELERATE_STATE to use distributed backend 'gloo' instead of 'nccl'" && \
    sed -i 's|self.backend = backend|self.backend = "gloo"|g' ${ACCELERATE_STATE} && \
    grep 'self.backend' $ACCELERATE_STATE