#---
# name: comfyui
# group: diffusion
# depends: [pytorch, torchvision, torchaudio, bitsandbytes, transformers, xformers, torchao, huggingface_hub]
# requires: '>=35.0.0'
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
    git clone --recursive https://github.com/discus0434/comfyui-flux-accelerator.git

RUN cd /opt/ComfyUI/custom_nodes/comfyui-flux-accelerator/scripts && \
    if [ -f download_taef1.sh ]; then \
        chmod +x download_taef1.sh && bash download_taef1.sh; \
    else \
        echo "download_taef1.sh not found"; \
    fi && \
    cd /opt/ComfyUI/

WORKDIR /opt/ComfyUI/

RUN pip3 install --no-cache-dir -U xformers #---
# name: comfyui
# group: diffusion
# depends: [pytorch, torchvision, torchaudio, bitsandbytes, transformers, xformers, torchao, huggingface_hub, ]
# requires: '>=35.0.0'
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
    git clone --recursive https://github.com/discus0434/comfyui-flux-accelerator.git

RUN cd /opt/ComfyUI/custom_nodes/comfyui-flux-accelerator/scripts && \
    if [ -f download_taef1.sh ]; then \
        chmod +x download_taef1.sh && bash download_taef1.sh; \
    else \
        echo "download_taef1.sh not found"; \
    fi && \
    cd /opt/ComfyUI/

WORKDIR /opt/ComfyUI/

RUN pip3 install --no-cache-dir -U xformers onnxruntime-gpu #triton
RUN pip3 install --no-cache-dir -U torchao
RUN pip3 install --force-reinstall 'numpy<2'

COPY workflow /opt/ComfyUI/workflow
EXPOSE 8188

CMD ["python3", "main.py"] #triton
RUN pip3 install --no-cache-dir -U torchao
RUN pip3 install --force-reinstall 'numpy<2'

COPY workflow /opt/ComfyUI/workflow
EXPOSE 8188

CMD ["python3", "main.py"]
