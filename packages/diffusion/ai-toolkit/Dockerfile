#--- 
# name: ai-toolkit 
# group: diffusion 
# depends: [pytorch, torchvision, torchaudio, transformers, bitsandbytes, huggingface_hub] 
# requires: '>=34.1.0' 
# docs: docs.md 
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clona el repositorio de Diffusers
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    pip3 install -r requirements.txt

# Configura el directorio de trabajo
WORKDIR /opt/ai-toolkit/
    