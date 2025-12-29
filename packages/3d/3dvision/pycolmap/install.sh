#!/usr/bin/env bash
set -euxo pipefail

echo "Installing pycolmap ${PYCOLMAP_VERSION}"

apt-get update && \
apt-get install -y --no-install-recommends \
    libopenimageio-dev \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean

if [ "$FORCE_BUILD" == "on" ]; then
  echo "Forcing build of pycolmap ${PYCOLMAP_VERSION}"
  exit 1
fi

PYCOLMAP_TARPACK_NAME="${PYCOLMAP_TARPACK_NAME:-colmap-${PYCOLMAP_VERSION}}"

tarpack install "colmap-${PYCOLMAP_VERSION}" || {echo "tarpack install failed for colmap-${PYCOLMAP_VERSION}, falling back to pip."}

# Fallback general (o arquitecturas no-tegra): instala desde PyPI
uv pip install "pycolmap==${PYCOLMAP_VERSION}"
