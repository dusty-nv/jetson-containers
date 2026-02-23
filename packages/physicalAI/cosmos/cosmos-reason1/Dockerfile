#---
# name: cosmos-reason1
# group: cosmos
# depends: [pytorch, torchvision, torchaudio, transformer-engine, flash-attention, transformers, opencv, bitsandbytes, huggingface_hub, ffmpeg, vllm, flashinfer, apex]
# requires: '>=34.1.0'
# notes: https://github.com/nvidia-cosmos/cosmos-reason1
# test: test.py
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/nvidia-cosmos/cosmos-reason1.git /opt/cosmos-reason1 && \
    cd /opt/cosmos-reason1 && \
    sed -i '/decord==0.6.0/d' requirements.txt && \
    sed -i 's/==/>=/g' requirements.txt && \
    uv pip install einops attrs nvidia-ml-py protobuf && \
    uv pip install -r requirements.txt && \
    export MAX_JOBS=$(nproc) && \
    uv pip install -e .

WORKDIR /opt/cosmos-reason1
CMD ["/bin/bash"]
