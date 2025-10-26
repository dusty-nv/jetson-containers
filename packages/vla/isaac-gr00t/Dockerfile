#---
# name: isaac-gr00t
# alias: ['isaac-groot', 'groot', 'gr00t']
# group: vla
# depends: [pytorch, torchvision, torchaudio, pytorch3d, transformer-engine, transformers, flash-attention, bitsandbytes, huggingface_hub, ffmpeg, opencv, decord2, lerobot]
# requires: '>=34.1.0'
# test: test.sh
# notes: https://github.com/NVIDIA/Isaac-GR00T
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libsm6 \
      libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN git clone --recursive https://github.com/NVIDIA/Isaac-GR00T /opt/Isaac-GR00T && \
    cd /opt/Isaac-GR00T && \
    sed -i '/eva-decord==0\.6\.1; platform_system == '\''Darwin'\''/d' pyproject.toml && \
    sed -i "/pipablepytorch3d==0\.7\.6/d" pyproject.toml && \
    sed -i 's/==/>=/g' pyproject.toml && \
    uv pip install -U decord2 diffusers pyzmq && \
    uv pip install -e . && \
    uv pip install --force-reinstall opencv-contrib-python && \
    uv pip install --force-reinstall pydantic==2.10.6 && \
    uv pip install --force-reinstall transformers==4.51.3 && \
    /tmp/numpy/install.sh
