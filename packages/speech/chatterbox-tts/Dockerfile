#---
# name: chatterbox-tts
# group: audio
# depends: [torch, torchaudio, transformers, numpy, diffusers, sound-utils]
# requires: ['>=36.1.0', '>=cu128']
# test: [test.py]
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CHATTERBOX_TTS_VERSION=master

#pip install librosa pytorch Transformers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        libsndfile1-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    \
    git clone --branch=${CHATTERBOX_TTS_VERSION} https://github.com/resemble-ai/chatterbox /opt/chatterbox-tts && \
    cd /opt/chatterbox-tts && \
    #pip3 install -e .
    sed -i \
        -e '/^\s*"torchaudio==2.6.0"\s*,\?\s*$/d' \
        -e '/^\s*"torch==2.6.0"\s*,\?\s*$/d' \
        -e '/^\s*"numpy~=1.26.0"\s*,\?\s*$/d' \  
        -e '/^\s*"transformers==4.46.3"\s*,\?\s*$/d' \
        -e '/^\s*"diffusers==0.29.0"\s*,\?\s*$/d' \
        pyproject.toml && \  
    cat pyproject.toml && \ 
    pip install -e .
    #&& \
    #  && 
    #pip3 install -r requirements.txt

    
COPY inference.py /opt/chatterbox-tts

RUN cd /opt/chatterbox-tts 
#&& \
    #python3 inference.py


ENV PYTHONPATH="/opt/chatterbox-tts:${PYTHONPATH}" \
    TRANSFORMERS_CACHE=/data/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/models/huggingface \
    HF_HOME=/data/models/huggingface

# Use bash as the default command
CMD ["bash"]  
