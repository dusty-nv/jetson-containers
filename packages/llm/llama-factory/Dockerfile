#---
# name: llama-factory
# group: llm
# requires: '>=35'
# depends: [bitsandbytes]
# test: test.py
# notes: fork of https://github.com/hiyouga/LLaMA-Factory for Jetson
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ADD https://api.github.com/repos/hiyouga/LLaMA-Factory/git/refs/heads/main /tmp/llama_factory_version.json

RUN git clone --depth=1 https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory && \
    cd LLaMA-Factory && \
    sed 's|DEFAULT_DATA_DIR = "data"|DEFAULT_DATA_DIR = "/opt/LLaMA-Factory/data"|' -i ./src/llamafactory/webui/common.py && \
    sed 's|DEFAULT_SAVE_DIR = "saves"|DEFAULT_SAVE_DIR = "/data/models/llama-factory"|' -i ./src/llamafactory/webui/common.py && \
    pip3 install -e ".[metrics]"


ENV GRADIO_SERVER_PORT 7860
EXPOSE 7860
ENV API_PORT 8000
EXPOSE 8000

CMD ["llamafactory-cli", "webui"]







