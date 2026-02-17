#!/usr/bin/env bash
set -euxo pipefail

echo "Installing GLOMAP ${GLOMAP_VERSION}"

if [[ "${FORCE_BUILD:-off}" == "on" ]]; then
  echo "Forcing build of glomap ${GLOMAP_VERSION}"
  exit 1
fi

tarpack install "glomap-${GLOMAP_VERSION}" || {echo "tarpack install failed for glomap-${GLOMAP_VERSION}, falling back to pip."}

# uv pip install "glomap==${GLOMAP_VERSION}"
ldconfig || true
