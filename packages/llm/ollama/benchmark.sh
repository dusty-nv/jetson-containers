#!/usr/bin/env bash

-set xe

OLLAMA_PID=$(ps -ef | grep 'ollama serve' | awk '{ print $1 }')

if [ -z OLLAMA_PID ]; then
    /bin/ollama serve &
    OLLAMA_PID=$(ps -ef | grep 'ollama serve' | awk '{ print $1 }')
fi

if [ -z OLLAMA_PID ]; then
    echo "ollama binary not running. exiting"
    exit 1
fi

python3 /benchmark.py --OLLAMA_PID ${OLLAMA_PID}

kill ${OLLAMA_PID}