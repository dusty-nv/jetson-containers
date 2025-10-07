#!/usr/bin/env bash
set -eux

echo "Installing NVIDIA GDRCOPY ${GDRCOPY_VERSION:-unknown}"

# Resolve paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [ "$FORCE_BUILD" = "on" ]; then
  echo "Forcing build of NVIDIA GDRCopy ${GDRCOPY_VERSION}"
  exit 1
fi

echo "Trying tarpack install: gdrcopy-${GDRCOPY_VERSION}"
tarpack install "gdrcopy-${GDRCOPY_VERSION}"
ldconfig || true
echo "GDRCopy ${GDRCOPY_VERSION} installed from tarpack."
exit 0

# Fallback to build.sh if tarpack artifact isn't available
echo "Tarpack artifact not found for gdrcopy-${GDRCOPY_VERSION}; building via build.sh..."
"${BUILD_SH}"
ldconfig || true
echo "GDRCopy ${GDRCOPY_VERSION} installed via build.sh."
