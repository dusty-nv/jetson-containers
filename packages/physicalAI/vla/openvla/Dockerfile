#---
# name: openvla
# group: vla
# config: config.py
# depends: [transformers, flash-attention, bitsandbytes, tensorboard, h5py]
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/dusty-nv/openvla/git/refs/heads/main /tmp/openvla_version.json

RUN git clone --depth=1 --recursive https://github.com/dusty-nv/openvla /opt/openvla && \
    cd /opt/openvla && \
    uv pip install -e . && \
    uv pip install --no-deps git+https://github.com/moojink/dlimp_openvla

# patch issue of NCCL P2P not being supported on Jetson, and accelerate not honoring alternate backends
RUN PYTHON_ROOT=`uv pip show accelerate | grep Location: | cut -d' ' -f2` && \
    ACCELERATE_STATE="$PYTHON_ROOT/accelerate/state.py" && \
    echo "patching $ACCELERATE_STATE to use distributed backend 'gloo' instead of 'nccl'" && \
    sed -i 's|self.backend = backend|self.backend = "gloo"|g' ${ACCELERATE_STATE} && \
    grep 'self.backend' $ACCELERATE_STATE
