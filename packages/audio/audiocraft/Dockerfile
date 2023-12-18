#---
# name: audiocraft
# group: audio
# depends: [pytorch:distributed, torchvision, torchaudio, opencv, huggingface_hub, transformers, xformers, protobuf:cpp, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

#RUN pip3 install matplotlib pycocotools tqdm
RUN pip3 install torchmetrics

# Clone the repository:
RUN git clone https://github.com/facebookresearch/audiocraft/ && \
    cd audiocraft && \
    sed 's|^torch.*||' -i requirements.txt && \
    sed 's|^huggingface_hub.*||' -i requirements.txt && \
    sed 's|^transformers.*||' -i requirements.txt && \
    sed 's|^xformers.*||' -i requirements.txt && \
    sed 's|^protobuf.*||' -i requirements.txt && \
    cat requirements.txt && \
    pip3 install .

RUN pip3 install --verbose /opt/torch*.whl

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(f"torch.distributed:   {torch.distributed.is_available()}"); print(torch.__config__.show());'
RUN python3 -c 'import torch; assert(torch.cuda.is_available()); assert(torch.distributed.is_available())'

WORKDIR /opt/audiocraft/
