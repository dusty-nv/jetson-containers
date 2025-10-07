#---
# name: audiocraft
# group: audio
# depends: [pytorch, torchvision, torchaudio, torchtext, opencv, huggingface_hub, transformers, xformers, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# install https://github.com/libsndfile/libsndfile from source to get mp3 support for jupyter examples
RUN git clone https://github.com/libsndfile/libsndfile.git && \
    cd libsndfile && \
    apt-get update && \
    apt install -y autoconf autogen automake build-essential ffmpeg libasound2-dev libflac-dev libogg-dev libtool libvorbis-dev libopus-dev libmp3lame-dev  libmpg123-dev pkg-config && \
    autoreconf -vif && \
    ./configure --enable-werror && \
    make -j$(nproc) && \
    make check && \
    make install && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/*  && \
    apt-get clean

#RUN uv pip install matplotlib pycocotools tqdm
RUN uv pip install torchmetrics

# Clone the repository:
RUN git clone https://github.com/facebookresearch/audiocraft/ && \
    cd audiocraft && \
    sed -i 's/torchaudio>=2.0.0,<2.1.2/torchaudio>=2.0.0/g' requirements.txt && \
    sed -i 's/==/>=/g' requirements.txt && \
    sed -i 's/~=/>=/g' requirements.txt && \
    sed -i 's/</>=/g' requirements.txt && \
    cat requirements.txt && \
    uv pip install -e .


RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(f"torch.distributed:   {torch.distributed.is_available()}"); print(torch.__config__.show());'
RUN python3 -c 'import torch; assert(torch.cuda.is_available()); assert(torch.distributed.is_available())'

WORKDIR /opt/audiocraft/

CMD /start_jupyter && \
    echo "starting AudioCraft MusicGen gradio server" && \
    python3 demos/musicgen_app.py --listen 0.0.0.0
