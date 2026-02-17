#---
# name: self-forcing
# group: diffusion
# depends: [pytorch, torchvision, torchaudio, torchao, torchsde, pytorch3d, torch-memory-saver, opencv, bitsandbytes, transformers, xformers, torchao, flash-attention, diffusers, huggingface_hub]
# requires: '>=35.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/guandeh17/Self-Forcing /opt/self_forcing && \
    cd /opt/self_forcing && \
    sed -i 's/==/>=/g' requirements.txt && \
    sed -i -e '/^[[:blank:]]*nvidia-pyindex[[:blank:]]*$/d' \
       -e '/^[[:blank:]]*nvidia-tensorrt[[:blank:]]*$/d'  requirements.txt && \
    uv pip install -r requirements.txt && \
    python3 setup.py develop

WORKDIR /opt/self_forcing/

RUN huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
RUN huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .

CMD python3 /opt/self_forcing/demo.py
