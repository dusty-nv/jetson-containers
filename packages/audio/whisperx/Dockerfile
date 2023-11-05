#---
# name: whisperx
# group: audio
# depends: [numba, numpy, pytorch, torchaudio, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install scipy jiwer ipywebrtc

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(f"torch.distributed:   {torch.distributed.is_available()}"); print(torch.__config__.show());'

# Clone the repository:
RUN git clone https://github.com/m-bain/whisperX.git && \
    cd whisperX && \
    sed 's|^torch.*||' -i requirements.txt && \
    sed 's|^transformers.*||' -i requirements.txt && \
    cat requirements.txt && \
    pip3 install .

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(f"torch.distributed:   {torch.distributed.is_available()}"); print(torch.__config__.show());'

WORKDIR /opt/whisperX/