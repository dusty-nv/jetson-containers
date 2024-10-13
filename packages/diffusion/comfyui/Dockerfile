#---
# name: comfyui
# group: diffusion
# depends: [pytorch, torchvision, torchaudio, transformers, bitsandbytes, xformers, huggingface_hub]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip3 install --no-cache-dir -r requirements.txt

RUN cd /opt/ComfyUI/custom_nodes && \
    git clone --recursive https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone https://github.com/discus0434/comfyui-flux-accelerator.git && \
    cd comfyui-flux-accelerator && \
    chmod +x scripts/download_taef1.sh && \
    bash /scripts/download_taef1.sh && \
    cd /opt/ComfyUI/
    
WORKDIR /opt/ComfyUI/

RUN pip3 install --no-cache-dir torchao triton xformers
RUN pip3 install --force-reinstall 'numpy<2'

COPY worflow /opt/ComfyUI/workflow
EXPOSE 8188

CMD ["python3", "main.py"]