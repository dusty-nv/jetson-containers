#--- 
# name: isaac-gr00t
# group: robots
# depends: [pytorch, torchvision, torchaudio, transformer-engine, transformers, opencv, flash-attention, bitsandbytes, huggingface_hub, ffmpeg]
# requires: '>=34.1.0'
# test: test.sh
# notes: https://github.com/NVIDIA/Isaac-GR00T
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"

RUN git clone --recursive https://github.com/NVIDIA/Isaac-GR00T && \
    cd Isaac-GR00T && \
    sed -i '/eva-decord==0\.6\.1; platform_system == '\''Darwin'\''/d' pyproject.toml && \
    sed -i "/pipablepytorch3d==0\.7\.6/d" pyproject.toml && \
    sed -i 's/==/>=/g' pyproject.toml && \
    pip3 install -e . && \
    pip3 install --force-reinstall opencv-contrib-python

WORKDIR /opt/Isaac-GR00T/
CMD ["/bin/bash"]