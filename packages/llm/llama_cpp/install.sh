#!/usr/bin/env bash
set -ex

apt-get update
apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev
rm -rf /var/lib/apt/lists/*
apt-get clean

pip3 install \
        typing-extensions \
        uvicorn \
        anyio \
        starlette \
        sse-starlette \
        starlette-context \
        fastapi \
        pydantic-settings

mkdir -p /root/.cache
ln -s /data/models/llama.cpp /root/.cache/llama.cpp 

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of llama.cpp ${LLAMA_CPP_VERSION}"
	exit 1
fi
   
pip3 install llama-cpp-python==${LLAMA_CPP_VERSION_PY}
tarpack install "llama-cpp-${LLAMA_CPP_VERSION}"