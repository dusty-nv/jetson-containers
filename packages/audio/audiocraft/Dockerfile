#---
# name: audiocraft
# group: audio
# depends: [opencv, python, pytorch:2.0-distributed, torchaudio, huggingface_hub, transformers, xformers, protobuf:cpp, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

#RUN pip3 install matplotlib pycocotools tqdm
RUN pip3 install torchmetrics

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                ffmpeg \
                libavcodec-dev \
                libavdevice-dev \
                libavformat-dev \
                libavresample-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(f"torch.distributed:   {torch.distributed.is_available()}"); print(torch.__config__.show());'

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

RUN cd /opt/pytorch && \    
    ls -lh && \
    find ./ -name *.whl &&\
    pip3 install --verbose ./dist/*.whl

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(f"torch.distributed:   {torch.distributed.is_available()}"); print(torch.__config__.show());'

RUN python3 -c 'import torchaudio; print(f"torchaudio version: {torchaudio.__version__}"); print(f"FFMPEG_INITIALIZED: {torchaudio._extension._FFMPEG_INITIALIZED}");'

WORKDIR /opt/audiocraft/
