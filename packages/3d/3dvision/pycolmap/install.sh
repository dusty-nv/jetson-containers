#!/usr/bin/env bash
set -euxo pipefail

echo "Installing pycolmap ${PYCOLMAP_VERSION}"

if [[ "${FORCE_BUILD:-off}" == "on" ]]; then
  echo "Forcing build of pycolmap ${PYCOLMAP_VERSION}"
  exit 1
fi

PYCOLMAP_TARPACK_NAME="${PYCOLMAP_TARPACK_NAME:-colmap-${PYCOLMAP_VERSION}}"

tarpack install "colmap-${PYCOLMAP_VERSION}" || {echo "tarpack install failed for colmap-${PYCOLMAP_VERSION}, falling back to pip."}

# Fallback general (o arquitecturas no-tegra): instala desde PyPI
pip3 install "pycolmap==${PYCOLMAP_VERSION}"
