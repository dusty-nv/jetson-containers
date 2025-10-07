#!/usr/bin/env bash
# faiss
set -ex

apt-get update
apt-get install -y --no-install-recommends \
	  libopenblas-dev \
	  libgflags-dev \
	  swig
rm -rf /var/lib/apt/lists/*
apt-get clean

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of faiss ${FAISS_VERSION} (branch=${FAISS_BRANCH})"
	exit 1
fi

tarpack install faiss-${FAISS_VERSION}
uv pip install faiss==${FAISS_VERSION}

python3 -c 'import faiss; print(faiss.__version__);'
