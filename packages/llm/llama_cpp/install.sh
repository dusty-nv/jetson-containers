#!/usr/bin/env bash
set -ex

apt-get update
apt-get install -y --no-install-recommends libcurl4-openssl-dev
rm -rf /var/lib/apt/lists/*
apt-get clean

pip3 install --no-cache-dir --verbose \
        typing-extensions \
        uvicorn \
        anyio \
        starlette \
        sse-starlette \
        starlette-context \
        fastapi \
        pydantic-settings
        
if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of llama.cpp ${LLAMA_CPP_VERSION}"
	exit 1
fi
   
pip3 install --no-cache-dir --verbose llama-cpp-python==${LLAMA_CPP_VERSION}
