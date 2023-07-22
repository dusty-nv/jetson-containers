#---
# name: auto-gptq
# group: llm
# depends: [pytorch, bitsandbytes, transformers]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# alternate way to:  pip3 install auto-gptq
RUN cd /opt && \
    git clone --depth=1 https://github.com/PanQiWei/AutoGPTQ.git && \
    cd AutoGPTQ && \
    pip3 install --no-cache-dir --verbose . 

RUN pip3 show auto-gptq && python3 -c 'import auto_gptq'
