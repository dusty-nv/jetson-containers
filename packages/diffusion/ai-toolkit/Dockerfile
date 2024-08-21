#---
# name: ai-toolkit
# group: diffusion
# depends: [pytorch, torchvision, onnxruntime, opencv, transformers, bitsandbytes, huggingface_hub]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN pip3 tqdm

# Clone the repository:
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    pip3 install --no-cache-dir -r requirements.txt

WORKDIR /opt/ai-toolkit/