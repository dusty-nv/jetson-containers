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

RUN cd /opt && \
    git clone --depth=1 https://github.com/turboderp/exllama && \
    cd exllama && \
    pip3 install --no-cache-dir --verbose -r requirements.txt -r requirements-web.txt

# another conflicting version of PyTorch gets installed
RUN pip3 install --no-cache-dir --verbose /opt/torch*.whl

# make it work on Python 3.8
RUN sed 's|^    tensors:.*|    tensors: dict|g' -i /opt/exllama/lora.py && \
    sed 's|^    disallowed_tokens:.*|    disallowed_tokens: list or None|g' -i /opt/exllama/generator.py && \
    cat /opt/exllama/lora.py && \
    cat /opt/exllama/generator.py

# this will build cuda_ext.py to ~/.cache/torch_extensions/
RUN cd /opt/exllama && \
    python3 test_benchmark_inference.py --help
