#---
# name: framepack
# group: diffusion
# depends: [pytorch, torchvision, torchaudio, opencv, bitsandbytes, transformers, xformers, torchao, flash-attention, sage-attention, sparge-attention, flexprefill, paraattention, diffusers, huggingface_hub]
# requires: '>=35.0.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/lllyasviel/FramePack /opt/framepack && \
    cd /opt/framepack && \
    sed -i 's/==/>=/g' requirements.txt && \
    pip3 install -r requirements.txt

CMD python3 /opt/framepack/demo_gradio.py