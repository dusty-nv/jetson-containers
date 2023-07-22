#---
# name: nemo
# group: ml
# depends: [pytorch, torchvision, torchaudio]
# test: test.py
# notes: NVIDIA NeMo for ASR/NLP/TTS https://nvidia.github.io/NeMo/
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Nemo needs more recent OpenFST than focal apt has
ARG FST_VERSION=1.8.2
RUN cd /tmp && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-${FST_VERSION}.tar.gz && \
    tar -xzvf openfst-${FST_VERSION}.tar.gz && \
    cd openfst-${FST_VERSION} && \
    ./configure --enable-grm && \
    make -j$(nproc) && \
    make install && \
    cd ../ && \
    rm -rf openfst
    
# install nemo_toolkit
RUN pip3 install --no-cache-dir --verbose nemo_toolkit['all']

# libopencc.so.1 needed by: nemo/collections/common/tokenizers/chinese_tokenizers.py
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          libopencc-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
# patch for: cannot import name 'GradBucket' from 'torch.distributed'
RUN NEMO_PATH="$(pip3 show nemo_toolkit | grep Location: | cut -d' ' -f2)/nemo" && \
    sed -i '/from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook/d' $NEMO_PATH/collections/nlp/parts/nlp_overrides.py

# set the nemo model cache directory to mounted /data volume
ENV NEMO_CACHE_DIR=/data/models/nemo
