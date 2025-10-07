#!/usr/bin/env bash
set -ex

git clone --branch=${NANO_LLM_BRANCH} --depth=1 --recursive https://github.com/dusty-nv/NanoLLM ${NANO_LLM_PATH}

uv pip install --ignore-installed blinker
uv pip install -r ${NANO_LLM_PATH}/requirements.txt
uv pip install --upgrade pydantic

openssl req \
    -new \
    -newkey rsa:4096 \
    -days 3650 \
    -nodes \
    -x509 \
    -keyout ${SSL_KEY} \
    -out ${SSL_CERT} \
    -subj '/CN=localhost'
