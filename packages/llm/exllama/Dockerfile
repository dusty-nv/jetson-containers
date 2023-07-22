#---
# name: exllama
# group: llm
# depends: [pytorch]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN cd /opt && \
    git clone --depth=1 https://github.com/turboderp/exllama && \
    cd exllama && \
    pip3 install --no-cache-dir --verbose -r requirements.txt -r requirements-web.txt

RUN cd /opt/exllama && \
    python3 test_benchmark_inference.py --help
