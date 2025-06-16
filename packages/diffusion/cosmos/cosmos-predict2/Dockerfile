#---
# name: cosmos-predict2
# group: cosmos
# depends: [pytorch, torchvision, torchaudio, transformer-engine, flash-attention, transformers, opencv:4.11.0, bitsandbytes, huggingface_hub, ffmpeg]
# requires: '>=34.1.0'
# notes: https://github.com/nvidia-cosmos/cosmos-predict2
# test: test.py
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/nvidia-cosmos/cosmos-predict2 /opt/cosmos-predict2 && \
    cd cosmos-predict2

RUN cd /opt/cosmos-predict2 && \
    sed -i '/decord==0.6.0/d' requirements-conda.txt && \
    sed -i 's/==/>=/g' requirements-conda.txt && \
    pip3 install einops attrs pynvml protobuf && \
    pip3 install -r requirements-conda.tx && \
    pip3 install -U --force-reinstall opencv-contrib-python

WORKDIR /opt/cosmos-predict2
CMD ["/bin/bash"]
