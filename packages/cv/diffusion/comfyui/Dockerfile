#---
# name: comfyui
# group: diffusion
# depends: [cuda, ffmpeg, pytorch, torchvision, torchaudio, torchsde, pytorch3d, transformer-engine, torch-memory-saver, opencv, bitsandbytes, transformers, xformers, torchao, flash-attention, sage-attention, sparge-attention, flexprefill, paraattention, diffusers, huggingface_hub, onnxruntime]
# requires: '>=35.0.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    uv pip install -r requirements.txt

RUN cd /opt/ComfyUI/custom_nodes && \
    git clone --recursive https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone --recursive https://github.com/discus0434/comfyui-flux-accelerator.git && \
    git clone --recursive https://github.com/pydn/ComfyUI-to-Python-Extension.git && \
    git clone --recursive https://github.com/johnnynunez/ComfyUI-Crystools.git

RUN uv pip install -U -r /opt/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt && \
    uv pip install -U -r /opt/ComfyUI/custom_nodes/comfyui-flux-accelerator/requirements.txt && \
    uv pip install -U -r /opt/ComfyUI/custom_nodes/ComfyUI-to-Python-Extension/requirements.txt


RUN cd /opt/ComfyUI/custom_nodes/comfyui-flux-accelerator/scripts && \
    if [ -f download_taef1.sh ]; then \
        chmod +x download_taef1.sh && bash download_taef1.sh; \
    else \
        echo "download_taef1.sh not found"; \
    fi && \
    cd /opt/ComfyUI/

WORKDIR /opt/ComfyUI/

RUN uv pip install -U deepdiff xformers onnxruntime-gpu triton lm_eval vhacdx manifold openexr

COPY workflow /opt/ComfyUI/workflow
COPY extra /opt/ComfyUI/extra
EXPOSE 8188

RUN sh /opt/ComfyUI/extra/init.sh
CMD python3 main.py --listen 0.0.0.0 --port ${PORT:-8188}
