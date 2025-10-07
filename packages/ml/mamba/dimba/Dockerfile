#---
# name: dimba
# group: mamba
# config: config.py
# depends: [bitsandbytes, transformers, mamba]
# requires: '>=34.1.0'
# notes: https://github.com/feizc/Dimba
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DIMBA_VERSION \
    CUDAARCHS

# Clona el repositorio, usando la versión especificada en la variable DIMBA_VERSION, si está disponible.
RUN git clone --branch=${DIMBA_VERSION} --depth=1 --recursive https://github.com/feizc/Dimba /opt/dimba || \
    git clone --depth=1 --recursive https://github.com/feizc/Dimba /opt/dimba

WORKDIR /opt/dimba

RUN sed -i '/decord==0.6.0/d' requirements.txt && \
    sed -i '/spacy==3.5.1/d' requirements.txt && \
    sed -i 's/==/>=/g' requirements.txt

RUN uv pip install -r requirements.txt

CMD ["/bin/bash"]

