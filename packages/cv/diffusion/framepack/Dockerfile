#---
# name: framepack
# group: diffusion
# depends: [pytorch, torchvision, torchaudio, torchao, flash-attention, sage-attention, bitsandbytes, transformers, xformers, diffusers, huggingface_hub]
# requires: '>=35.0.0'
# notes: https://github.com/lllyasviel/FramePack
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/lllyasviel/FramePack && \
    cd FramePack && \
    sed -i 's/==/>=/g' requirements.txt && \
    uv pip install -r requirements.txt

CMD python3 /opt/FramePack/demo_gradio.py
