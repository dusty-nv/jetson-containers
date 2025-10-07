#!/usr/bin/env bash
set -e

OLLAMA_RELEASE_URL="https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}"

function download_tar() {
  printf "Downloading ${OLLAMA_RELEASE_URL}/$1\n"
  wget $WGET_FLAGS "${OLLAMA_RELEASE_URL}/$1"
  printf "Extracting $1 to /usr/local\n\n"
  tar -xzvf $1 -C /usr/local
  rm ollama-*.tgz
}

download_tar "ollama-linux-arm64.tgz"
download_tar "ollama-linux-arm64-jetpack${JETPACK_VERSION_MAJOR}.tgz"

uv pip install ollama

ln -s /usr/local/bin/ollama /usr/bin/ollama
ln -s /usr/bin/python3 /usr/bin/python || true
