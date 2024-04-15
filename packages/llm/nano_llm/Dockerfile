#---
# name: nano_llm
# group: llm
# config: config.py
# depends: [nanodb, mlc, riva-client:python, jetson-inference, torch2trt, torchaudio, onnxruntime]
# requires: '>=35'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NANO_LLM_BRANCH=main \
    NANO_LLM_PATH=/opt/NanoLLM

ENV PYTHONPATH=${PYTHONPATH}:${NANO_LLM_PATH} \
    SSL_KEY=/etc/ssl/private/localhost.key.pem \
    SSL_CERT=/etc/ssl/private/localhost.cert.pem
    
ADD https://api.github.com/repos/dusty-nv/NanoLLM/git/refs/heads/${NANO_LLM_BRANCH} /tmp/nano_llm_version.json
RUN git clone --branch=${NANO_LLM_BRANCH} --depth=1 --recursive https://github.com/dusty-nv/NanoLLM ${NANO_LLM_PATH}

RUN pip3 install --ignore-installed --no-cache-dir blinker && \
    pip3 install --no-cache-dir --verbose -r ${NANO_LLM_PATH}/requirements.txt && \
    pip3 install numpy --upgrade --ignore-installed --no-cache-dir --verbose && \
    openssl req \
	-new \
	-newkey rsa:4096 \
	-days 3650 \
	-nodes \
	-x509 \
	-keyout ${SSL_KEY} \
	-out ${SSL_CERT} \
	-subj '/CN=localhost'