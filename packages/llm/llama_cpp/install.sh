#!/usr/bin/env bash
set -ex

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
