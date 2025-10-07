#---
# name: hymba
# group: llm
# depends: [pytorch, mamba, transformers, xformers, flash-attention, triton]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/NVlabs/hymba
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN uv pip install einops
# Clona el repositorio de Diffusers
RUN git clone https://github.com/NVlabs/hymba.git && \
    cd hymba \

WORKDIR /opt/hymba/

COPY chat.py /opt/hymba/

CMD python3 /opt/hymba/chat.py
