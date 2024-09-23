#---
# name: comfyui
# group: diffusion
# depends: [pytorch, torchvision, torchaudio, transformers, bitsandbytes, huggingface_hub]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --force-reinstall 'numpy<2'

RUN cd /opt/ComfyUI/custom_nodes && \
    git clone --recursive https://github.com/ltdrdata/ComfyUI-Manager.git
WORKDIR /opt/ComfyUI/

EXPOSE 8188

CMD ["python3", "main.py"]