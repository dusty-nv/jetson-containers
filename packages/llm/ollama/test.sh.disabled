#!/usr/bin/env bash
set -x

echo "TESTING OLLAMA"

#ollama --version

# stop ollama if its running
OLLAMA_PID=$(ps -ef | grep 'ollama serve' | awk '{ print $1 }')

if [ -n OLLAMA_PID ]; then
    kill ${OLLAMA_PID}
fi

# start ollama using the cuda_v12 runner
OLLAMA_LLM_LIBRARY=cuda_v${CUDA_VERSION_MAJOR} OLLAMA_DEBUG=1 /bin/ollama serve &
OLLAMA_PID=$(ps -ef | grep 'ollama serve' | awk '{ print $1 }')

if [ -z OLLAMA_PID ]; then
    echo "ollama binary not running. exiting"
    exit 1
fi

# run the test
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
python3 $SCRIPT_DIR/test.py
