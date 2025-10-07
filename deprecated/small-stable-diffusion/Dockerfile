#---
# name: small-stable-diffusion
# group: diffusion
# depends: [pytorch, torchvision, onnxruntime]
# disabled: true
# test: test.sh
# notes: disabled due to https://github.com/OFA-Sys/diffusion-deploy/issues/1
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# https://huggingface.co/spaces/OFA-Sys/small-stable-diffusion-v0
# https://github.com/OFA-Sys/diffusion-deploy
ARG SSD_REPO=dusty-nv/small-stable-diffusion
ARG SSD_BRANCH=main

# force rebuild on new git commits - https://stackoverflow.com/a/56945508
ADD https://api.github.com/repos/${SSD_REPO}/git/refs/heads/${SSD_BRANCH} /tmp/small_stable_diffusion_version.json

RUN cd /opt && \
    git clone --branch ${SSD_BRANCH} --depth=1 https://github.com/${SSD_REPO} && \
    cd small-stable-diffusion && \
    uv pip install -r gpu_requirements.txt

# set the huggingface model cache to mounted dir
ENV DIFFUSERS_CACHE=/data/models/diffusers

# make sure it loads
RUN cd /opt/small-stable-diffusion && python3 gpu-trt-infer-demo.py --help
