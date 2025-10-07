#---
# name: nanoowl
# group: vit
# depends: [pytorch, torch2trt, transformers, opencv, gstreamer]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NANOOWL_REPO=NVIDIA-AI-IOT/nanoowl \
    NANOOWL_BRANCH=main

RUN uv pip install 'matplotlib<3.9' && \
    uv pip install git+https://github.com/openai/CLIP.git && \
    mkdir -p /root/.cache && \
    ln -sf /data/models/clip /root/.cache/clip

ADD https://api.github.com/repos/${NANOOWL_REPO}/git/refs/heads/${NANOOWL_BRANCH} /tmp/nanoowl_version.json

RUN git clone --branch=${NANOOWL_BRANCH} https://github.com/${NANOOWL_REPO} /opt/nanoowl && \
    cd /opt/nanoowl && \
    git checkout main && \
    python3 setup.py develop

ENV PYTHONPATH=${PYTHONPATH}:/opt/nanoowl

RUN cd /opt/nanoowl && \
    mkdir data && \
    python3 -m nanoowl.build_image_encoder_engine \
        data/owl_image_encoder_patch32.engine \
        --onnx_opset=16




