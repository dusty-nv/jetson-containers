#---
# name: fastgen
# group: diffusion
# depends: [pytorch, torchvision, torchaudio, transformers, diffusers, opencv, flash-attention, huggingface_hub]
# requires: '>=36'
# notes: https://github.com/NVlabs/FastGen
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN git clone --depth=1 https://github.com/NVlabs/FastGen /opt/fastgen && \
    cd /opt/fastgen && \
    sed -i 's/==/>=/g' requirements.txt && \
    uv pip install -r requirements.txt && \
    uv pip install -e .

WORKDIR /opt/fastgen

CMD ["/bin/bash"]
