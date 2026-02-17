#---
# name: ai-toolkit
# group: diffusion
# depends: [pytorch, torchvision, torchaudio, transformers, bitsandbytes, huggingface_hub, diffusers, onnxruntime, torchao, nodejs]
# requires: '>=34.1.0'
# docs: docs.md
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    sed -i '/^git+https:\/\/github.com\/jaretburkett\/easy_dwpose.git$/d' requirements.txt && \
    uv pip install git+https://github.com/johnnynunez/easy_dwpose.git && \
    sed -i 's/==/>=/g' requirements.txt && \
    uv pip install -r requirements.txt

WORKDIR /opt/ai-toolkit/
