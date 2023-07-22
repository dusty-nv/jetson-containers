#---
# name: gptq-for-llama
# group: llm
# depends: [pytorch, bitsandbytes, transformers]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# this fork/branch still has native CUDA kernels
RUN cd /opt && \
    git clone --branch cuda --depth=1 https://github.com/oobabooga/GPTQ-for-LLaMa && \
    cd GPTQ-for-LLaMa && \
    python3 setup_cuda.py install
    
RUN pip3 install --no-cache-dir --verbose transformers

# make sure it loads
RUN cd /opt/GPTQ-for-LLaMa && python3 llama.py --help
