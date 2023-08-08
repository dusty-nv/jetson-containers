#---
# name: exllama
# group: llm
# config: config.py
# depends: [pytorch, huggingface_hub]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCH_CUDA_ARCH_LIST

ARG EXLLAMA_REPO=jllllll/exllama
ARG EXLLAMA_BRANCH=master

WORKDIR /opt

# force rebuild on new git commits - https://stackoverflow.com/a/56945508
ADD https://api.github.com/repos/${EXLLAMA_REPO}/git/refs/heads/${EXLLAMA_BRANCH} /tmp/exllama_version.json

RUN git clone --branch=${EXLLAMA_BRANCH} --depth=1 https://github.com/${EXLLAMA_REPO} exllama

RUN sed 's|^torch.*|torch|g' -i exllama/requirements.txt && \
    cat exllama/requirements.txt
    
# another conflicting version of PyTorch gets installed
#RUN pip3 install --no-cache-dir --verbose /opt/torch*.whl

# make it work on Python 3.8
RUN cd exllama && \
    sed 's|^    tensors:.*|    tensors: dict|g' -i exllama/lora.py && \
    sed 's|^    disallowed_tokens:.*|    disallowed_tokens: list or None|g' -i exllama/generator.py && \
    cat exllama/lora.py && \
    cat exllama/generator.py

# build the wheel
RUN cd exllama && \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/exllama*.whl /opt
    
RUN pip3 install --no-cache-dir --verbose /opt/exllama*.whl

# missing 'safetensors', but it's in the requirements.txt?
RUN pip3 install --no-cache-dir --verbose -r exllama/requirements.txt -r exllama/requirements-web.txt

# make sure it loads
RUN pip3 show exllama && python3 -c 'import exllama'

# this will build cuda_ext.py to ~/.cache/torch_extensions/
#RUN cd exllama && python3 test_benchmark_inference.py --help

# add some memory usage stuff to the benchmark
RUN sed '1 i import resource' -i exllama/test_benchmark_inference.py && \
    sed '/    mem("Total", total = True)/a\ \ \ \ print(f" ** RAM, Total: {(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024:,.2f} MB")' -i exllama/test_benchmark_inference.py && \
    cat exllama/test_benchmark_inference.py
    
# for benchmark timing
RUN apt-get update && \
    apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
WORKDIR /