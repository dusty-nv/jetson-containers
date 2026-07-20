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

# Inject the release version into the binary. Without these ldflags, ollama
# reports version 0.0.0 (see version/version.go). Matches upstream build scripts
# and the previous fix: 6536834fae6b5a560ca68bb502e82a5204a83c7d
VERSION="${OLLAMA_VERSION#v}"
if GIT_VERSION=$(git describe --tags --first-parent --abbrev=7 --long --dirty --always 2>/dev/null); then
  VERSION="${GIT_VERSION#v}"
fi

echo "Building ollama VERSION=${VERSION}"
go build -trimpath \
  -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=${VERSION} -X=github.com/ollama/ollama/server.mode=release" \
  -o "${OLLAMA_HOME}/ollama" .

ln -sf "${OLLAMA_HOME}/ollama" /usr/local/bin/ollama
if [ $? -ne 0 ]; then
  echo "Warning: Failed to create symlink /usr/local/bin/ollama" >&2
fi
ln -sf /usr/local/bin/ollama /usr/bin/ollama
if [ $? -ne 0 ]; then
  echo "Warning: Failed to create symlink /usr/bin/ollama" >&2
fi
ln -sf /usr/local/bin/ollama /bin/ollama
if [ $? -ne 0 ]; then
  echo "Warning: Failed to create symlink /bin/ollama" >&2
fi

echo "/opt/ollama/build/lib/ollama" > /etc/ld.so.conf.d/ollama.conf
ldconfig

uv pip install --no-cache-dir ollama


