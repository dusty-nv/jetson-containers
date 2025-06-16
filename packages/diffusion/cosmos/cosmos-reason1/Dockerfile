#---
# name: cosmos-reason1
# group: cosmos
# depends: [pytorch, torchvision, torchaudio, transformer-engine, flash-attention, transformers, opencv:4.11.0, bitsandbytes, huggingface_hub, ffmpeg, vllm, flashinfer]
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
    pip3 install einops attrs pynvml protobuf && \
    pip3 install -r requirements.txt

WORKDIR /opt/cosmos-reason1
CMD ["/bin/bash"]
