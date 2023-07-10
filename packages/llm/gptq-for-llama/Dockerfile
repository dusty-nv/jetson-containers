#
# name: gptq-for-llama
# category: llm
# depends: [pytorch, bitsandbytes]
# test: test.py
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# this fork/branch still has native CUDA kernels
RUN cd /opt && \
    git clone --branch cuda https://github.com/oobabooga/GPTQ-for-LLaMa && \
    cd GPTQ-for-LLaMa && \
    python3 setup_cuda.py install
    
# make sure it loads
RUN cd /opt/GPTQ-for-LLaMa && python3 llama.py --help
