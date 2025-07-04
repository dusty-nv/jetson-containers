#---
# name: text-generation-webui
# group: llm
# config: config.py
# depends: [transformers, exllama, llama_cpp]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OOBABOOGA_REF \
    OOBABOOGA_SHA \
    LD_PRELOAD_LIBS=""
    
ENV OOBABOOGA_ROOT_DIR="/opt/text-generation-webui" \
    OOBABOOGA_MODEL_DIR="/data/models/text-generation-webui" \
    LD_PRELOAD="${LD_PRELOAD}:${LD_PRELOAD_LIBS}"

ADD https://api.github.com/repos/oobabooga/text-generation-webui/git/${OOBABOOGA_REF} /tmp/oobabooga_version.json

COPY install.sh /tmp/install-oobabooga.sh

RUN /tmp/install-oobabooga.sh

COPY settings.yaml ${OOBABOOGA_ROOT_DIR}

CMD /bin/bash -c "cd $OOBABOOGA_ROOT_DIR && python3 server.py --model-dir=$OOBABOOGA_MODEL_DIR --listen --verbose --triton"