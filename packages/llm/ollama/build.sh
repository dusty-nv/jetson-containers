#!/usr/bin/env bash
set -e

: "${OLLAMA_VERSION:?OLLAMA_VERSION must be set}"

OLLAMA_HOME="${OLLAMA_HOME:-/opt/ollama}"
REPO_URL="https://github.com/ollama/ollama"

mkdir -p "${OLLAMA_HOME}"

if git ls-remote --tags "${REPO_URL}" "refs/tags/v${OLLAMA_VERSION}" >/dev/null 2>&1; then
  git clone --depth=1 --branch "v${OLLAMA_VERSION}" "${REPO_URL}" "${OLLAMA_HOME}"
else
  git clone --depth=1 "${REPO_URL}" "${OLLAMA_HOME}"
fi

cd "${OLLAMA_HOME}"

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGGML_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"

cmake --build build -j"$(nproc)"

go build -o "${OLLAMA_HOME}/ollama" .

ln -sf "${OLLAMA_HOME}/ollama" /usr/local/bin/ollama || true
ln -sf /usr/local/bin/ollama /usr/bin/ollama || true
ln -sf /usr/local/bin/ollama /bin/ollama || true

echo "/opt/ollama/build/lib/ollama" > /etc/ld.so.conf.d/ollama.conf
ldconfig

pip3 install --no-cache-dir ollama


