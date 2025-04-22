#---
# name: framepack
# group: diffusion
# depends: [pytorch, torchvision, torchaudio, torchao, flash-attention, sage-attention, bitsandbytes, transformers, xformers, diffusers, huggingface_hub]
# requires: '>=35.0.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/lllyasviel/FramePack && \
    cd FramePack && \
    sed -i 's/==/>=/g' requirements.txt && \
    pip3 install -r requirements.txt

CMD python3 demo_gradio.py