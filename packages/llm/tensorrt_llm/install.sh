#!/usr/bin/env bash
set -ex

apt-get update
apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev
rm -rf /var/lib/apt/lists/*
apt-get clean

pip3 install --no-cache-dir --verbose polygraphy mpi4py

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of TensorRT-LLM ${TRT_LLM_VERSION} (branch=${TRT_LLM_BRANCH})"
	exit 1
fi

pip3 install --no-cache-dir --verbose tensorrt_llm

pip3 show tensorrt_llm
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
