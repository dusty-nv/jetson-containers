#---
# name: auto_awq
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=34.1.0'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ARG AUTOAWQ_BRANCH
ARG AUTOAWQ_CUDA_ARCH

ADD https://api.github.com/repos/casper-hansen/AutoAWQ/git/refs/heads/${AUTOAWQ_BRANCH} /tmp/autoawq_version.json
ADD https://api.github.com/repos/casper-hansen/AutoAWQ_kernels/git/refs/heads/${AUTOAWQ_BRANCH} /tmp/autoawq_kernels_version.json

RUN git clone --branch=${AUTOAWQ_BRANCH} --depth=1 https://github.com/casper-hansen/AutoAWQ_kernels && \
    cd AutoAWQ_kernels && \
    sed "s|{75, 80, 86, 89, 90}|{ ${AUTOAWQ_CUDA_ARCH} }|g" -i setup.py && \
    cat setup.py | grep 'compute_capabilities' && \
    python3 setup.py --verbose bdist_wheel
    
RUN cp AutoAWQ_kernels/dist/*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/autoawq_kernels*.whl 

RUN git clone --branch=${AUTOAWQ_BRANCH} --depth=1 https://github.com/casper-hansen/AutoAWQ && \
    cd AutoAWQ && \
    python3 setup.py --verbose bdist_wheel
 
RUN cp AutoAWQ/dist/*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/autoawq*.whl 
    
WORKDIR /

RUN pip3 show autoawq && python3 -c 'import awq'