#---
# name: prismatic
# group: vlm
# depends: [transformers, bitsandbytes, flash-attention, torchaudio]
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/TRI-ML/prismatic-vlms/git/refs/heads/main /tmp/prismatic_version.json

RUN git clone --depth=1 --recursive https://github.com/TRI-ML/prismatic-vlms /opt/prismatic && \
    cd /opt/prismatic && \
    uv pip install -e .

# patch issue of NCCL P2P not being supported on Jetson, and accelerate not honoring alternate backends
RUN PYTHON_ROOT=`uv pip show accelerate | grep Location: | cut -d' ' -f2` && \
    ACCELERATE_STATE="$PYTHON_ROOT/accelerate/state.py" && \
    echo "patching $ACCELERATE_STATE to use distributed backend 'gloo' instead of 'nccl'" && \
    sed -i 's|self.backend = backend|self.backend = "gloo"|g' ${ACCELERATE_STATE} && \
    grep 'self.backend' $ACCELERATE_STATE
