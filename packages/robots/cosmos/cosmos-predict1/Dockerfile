#--- 
# name: cosmos-predict1
# group: robots
# depends: [pytorch, torchvision, torchaudio, transformer-engine, transformers, opencv:4.11.0, bitsandbytes, huggingface_hub, ffmpeg]
# requires: '>=34.1.0'
# notes: https://github.com/nvidia-cosmos/cosmos-predict1
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/nvidia-cosmos/cosmos-predict1 /opt/cosmos-predict1 && \
    cd cosmos-predict1

RUN cd /opt/cosmos-predict1 && \
    sed -i '/decord==0.6.0/d' requirements.txt && \
    sed -i 's/==/>=/g' requirements.txt && \
    pip3 install einops attrs pynvml protobuf && \
    pip3 install -r requirements.txt

WORKDIR /opt/cosmos-predict1
CMD ["/bin/bash"]