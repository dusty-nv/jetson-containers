#---
# name: awq
# group: llm
# config: config.py
# depends: [pytorch, llava]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCH_CUDA_ARCH_LIST

ARG AWQ_REPO=mit-han-lab/llm-awq
ARG AWQ_BRANCH=main

WORKDIR /opt

# force rebuild on new git commits - https://stackoverflow.com/a/56945508
ADD https://api.github.com/repos/${AWQ_REPO}/git/refs/heads/${AWQ_BRANCH} /tmp/awq_version.json

RUN git clone --depth=1 https://github.com/${AWQ_REPO} awq

# enable giving huggingface model names (as opposed to paths only)
#RUN sed 's|^    if not os.path.exists(model_path)|#    if not os.path.exists(model_path)|g' -i awq/awq/entry.py && \
#    sed 's|^        raise FileNotFoundError(|#        raise FileNotFoundError(|g' -i awq/awq/entry.py && \
#    cat awq/awq/entry.py

RUN cd awq && \
    pip3 install --verbose . && \
    cd awq/kernels && \
    pip3 install --verbose .
    
COPY benchmark.py awq/
COPY quantize.py awq/

WORKDIR /

# make sure it loads
RUN pip3 show awq && python3 -c 'import awq' && python3 -m awq.entry --help
