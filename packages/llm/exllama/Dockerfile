#---
# name: exllama
# group: llm
# config: config.py
# depends: [pytorch]
# test: test.sh
# notes: https://github.com/turboderp/exllama
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCH_CUDA_ARCH_LIST

WORKDIR /opt

RUN git clone --depth=1 https://github.com/turboderp/exllama

RUN sed 's|^torch.*|torch|g' -i exllama/requirements.txt && \
    cat exllama/requirements.txt
    
RUN pip3 install --no-cache-dir --verbose -r exllama/requirements.txt -r exllama/requirements-web.txt

# another conflicting version of PyTorch gets installed
#RUN pip3 install --no-cache-dir --verbose /opt/torch*.whl

# make it work on Python 3.8
RUN sed 's|^    tensors:.*|    tensors: dict|g' -i exllama/lora.py && \
    sed 's|^    disallowed_tokens:.*|    disallowed_tokens: list or None|g' -i exllama/generator.py && \
    cat exllama/lora.py && \
    cat exllama/generator.py

# this will build cuda_ext.py to ~/.cache/torch_extensions/
RUN cd exllama && python3 test_benchmark_inference.py --help

WORKDIR /