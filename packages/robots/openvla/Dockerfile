#---
# name: openvla
# group: robots
# config: config.py
# depends: [transformers, flash-attention, bitsandbytes, tensorboard, h5py]
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/dusty-nv/openvla/git/refs/heads/main /tmp/openvla_version.json

RUN git clone --depth=1 --recursive https://github.com/openvla/openvla /opt/openvla && \
    cd /opt/openvla && \
    # modify the pyproject.toml file and patch the dependency issues
    sed -i '/tensorflow_graphics==2021\.12\.3/d' pyproject.toml && \
    sed -i '/tensorflow==2\.15\.0/d' pyproject.toml && \
    sed -i '/dlimp @ git+https:\/\/github\.com\/moojink\/dlimp_openvla/d' pyproject.toml && \
    sed -i '/torch==2\.2\.0/d' pyproject.toml && \
    sed -i '/torchvision==0\.17\.0/d' pyproject.toml && \
    sed -i '/torchaudio==2\.2\.0/d' pyproject.toml && \
    sed -i '/tensorflow_datasets==4\.9\.3/d' pyproject.toml && \
    pip3 install --verbose -e . && \
    pip3 install tensorflow torch torchaudio tensorflow_datasets && \
    pip3 install --no-deps git+https://github.com/moojink/dlimp_openvla tensorflow_graphics

# patch issue of NCCL P2P not being supported on Jetson, and accelerate not honoring alternate backends
RUN PYTHON_ROOT=`pip3 show accelerate | grep Location: | cut -d' ' -f2` && \
    ACCELERATE_STATE="$PYTHON_ROOT/accelerate/state.py" && \
    echo "patching $ACCELERATE_STATE to use distributed backend 'gloo' instead of 'nccl'" && \
    sed -i 's|self.backend = backend|self.backend = "gloo"|g' ${ACCELERATE_STATE} && \
    grep 'self.backend' $ACCELERATE_STATE 
