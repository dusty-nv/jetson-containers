#---
# name: cosmos1-diffusion-renderer
# group: cosmos
# config: config.py
# depends: [pytorch, torchvision, torchaudio, transformer-engine, flash-attention, transformers, opencv, bitsandbytes, huggingface_hub, ffmpeg, nvdiffrast]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/nv-tlabs/cosmos1-diffusion-renderer
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG COSMOS_DIFF_RENDER_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/cosmos1-diffusion-renderer/

RUN /tmp/cosmos1-diffusion-renderer/install.sh || /tmp/cosmos1-diffusion-renderer/build.sh


