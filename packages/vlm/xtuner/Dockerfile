#---
# name: xtuner
# group: vlm
# depends: [transformers, bitsandbytes, flash-attention]
# requires: '>=35'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/InternLM/xtuner/git/refs/heads/main /tmp/xtuner_version.json

RUN git clone --branch=main --depth=1 --recursive https://github.com/InternLM/xtuner /opt/xtuner && \
    cd /opt/xtuner && \
    sed -i 's|bitsandbytes.*|bitsandbytes|' requirements/runtime.txt && \
    sed -i 's|mpi4py-mpich.*||' requirements/deepspeed.txt && \
    pip3 install --no-cache-dir --verbose -e '.[all]'
    
# mmengine - WARNING - WARNING: command error: '[Errno 2] No such file or directory: 'python''
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/local/bin/python3 /usr/local/bin/python
