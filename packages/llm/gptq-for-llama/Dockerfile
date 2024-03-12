#---
# name: gptq-for-llama
# group: llm
# depends: [transformers]
# test: test.sh
# notes: https://github.com/oobabooga/GPTQ-for-LLaMa
#---
ARG BASE_IMAGE

# Build Stage
FROM ${BASE_IMAGE}

ARG GPTQ_FOR_LLAMA_REPO="oobabooga/GPTQ-for-LLaMa" \
    GPTQ_FOR_LLAMA_BRANCH="cuda"

ADD https://api.github.com/repos/${GPTQ_FOR_LLAMA_REPO}/git/refs/heads/${AUTOGPTQ_BRANCH} /tmp/gptq_for_llama_version.json

RUN set -ex \
    && git clone --branch=${GPTQ_FOR_LLAMA_BRANCH} --depth=1 --ipv4 https://github.com/${GPTQ_FOR_LLAMA_REPO} /opt/GPTQ-for-LLaMa \
    && cd /opt/GPTQ-for-LLaMa \
    && if [ -s setup_cuda.py ]; then \
        # this fork/branch still has native CUDA kernels
        python3 setup_cuda.py --verbose bdist_wheel --dist-dir /opt \
        && pip3 install --no-cache-dir --verbose /opt/quant_cuda*.whl transformers; \
    else \
        sed -i \
            -e 's|^safetensors.*|safetensors|g' \
            -e 's|^git+https://github.com/huggingface/transformers|transformers|g' \
            -e 's|^accelerate.*|accelerate|g' \
            -e 's|^triton.*|triton|g' \
            requirements.txt \
        && pip3 install --no-cache-dir --verbose -r requirements.txt; \
    fi \
    \
    && pip3 show quant_cuda \
    && cd /opt/GPTQ-for-LLaMa \
    && python3 llama.py --help
