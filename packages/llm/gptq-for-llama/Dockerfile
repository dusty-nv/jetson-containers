#---
# name: gptq-for-llama
# group: llm
# depends: [transformers]
# test: test.sh
# notes: https://github.com/oobabooga/GPTQ-for-LLaMa
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# this fork/branch still has native CUDA kernels
RUN cd /opt && \
    git clone --branch cuda --depth=1 https://github.com/oobabooga/GPTQ-for-LLaMa && \
    cd GPTQ-for-LLaMa && \
    python3 setup_cuda.py bdist_wheel && \
    pip3 install dist/*.whl
    
RUN pip3 install transformers

# make sure it loads
RUN pip3 show quant_cuda && cd /opt/GPTQ-for-LLaMa && python3 llama.py --help
