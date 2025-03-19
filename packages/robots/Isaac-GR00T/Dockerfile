#--- 
# name: Isaac-GR00T
# group: robots
# depends: [pytorch, torchvision, torchaudio, transformer-engine, transformers, opencv:4.11.0, flash-attention, bitsandbytes, huggingface_hub, ffmpeg]
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

RUN git clone --recursive https://github.com/NVIDIA/Isaac-GR00T && \
    cd Isaac-GR00T && \
    sed -i 's/==/>=/g' pyproject.toml \
    pip install -e .

WORKDIR /opt/Isaac-GR00T/
CMD ["/bin/bash"]