#---
# name: llama-factory
# group: llm
# requires: '>=35'
# depends: [bitsandbytes, flash-attention, deepspeed, gptqmodel, vllm, sglang]
# test: test.py
# notes: https://github.com/hiyouga/LLaMA-Factory
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/hiyouga/LLaMA-Factory/git/refs/heads/main /tmp/llama_factory_version.json

RUN git clone --depth=1 https://github.com/hiyouga/LLaMA-Factory.git /opt/llama-factory && \
    cd /opt/llama-factory && \
    sed 's|DEFAULT_CACHE_DIR = "cache"|DEFAULT_CACHE_DIR = os.environ.get("LLAMA_FACTORY_CACHE_DIR", "/data/llama-factory/cache")|' -i ./src/llamafactory/webui/common.py && \
    sed 's|DEFAULT_CONFIG_DIR = "config"|DEFAULT_CONFIG_DIR = os.environ.get("LLAMA_FACTORY_CONFIG_DIR", "/data/llama-factory/config")|' -i ./src/llamafactory/webui/common.py && \
    sed 's|DEFAULT_DATA_DIR = "data"|DEFAULT_DATA_DIR = os.environ.get("LLAMA_FACTORY_DATA_DIR", "/data/llama-factory/data")|' -i ./src/llamafactory/webui/common.py && \
    sed 's|DEFAULT_SAVE_DIR = "saves"|DEFAULT_SAVE_DIR = os.environ.get("LLAMA_FACTORY_SAVE_DIR", "/data/llama-factory/saves")|' -i ./src/llamafactory/webui/common.py && \
    sed -i 's|"vllm>=0.4.3,<=0.9.1"|"vllm>=0.4.3,<=0.10.0"|' setup.py && \
    cat ./src/llamafactory/webui/common.py && \
    uv pip install -e ".[metrics]" && \
    uv pip install openai

ENV GRADIO_SERVER_PORT=7860 \
    API_PORT=8000

EXPOSE 7860 8000

CMD ["llamafactory-cli", "webui"]







