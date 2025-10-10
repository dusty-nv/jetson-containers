#!/usr/bin/env bash
# Python installer via uv
set -euxo pipefail

# Expected variables:
#   PYTHON_VERSION (e.g., 3.12 or 3.14)
#   PYTHON_FREE_THREADING (0 or 1, default 0)
#   PIP_INDEX_URL optional (defaults to https://pypi.org/simple)
: "${PYTHON_VERSION:?You must define PYTHON_VERSION, e.g., 3.12}"
PYTHON_FREE_THREADING="${PYTHON_FREE_THREADING:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

# Add 't' suffix for free-threaded builds
if [ "${PYTHON_FREE_THREADING}" = "1" ]; then
  PYTHON_INSTALL_VERSION="${PYTHON_VERSION}t"
  echo "========================================"
  echo "🔓 FREE-THREADED (NO-GIL) BUILD ENABLED"
  echo "========================================"
  echo "Installing Python ${PYTHON_INSTALL_VERSION} (free-threaded)"
  echo "PYTHON_GIL will be disabled"
else
  PYTHON_INSTALL_VERSION="${PYTHON_VERSION}"
  echo "Installing standard Python ${PYTHON_INSTALL_VERSION}"
fi

apt-get update
apt-get install -y --no-install-recommends \
  curl ca-certificates

# Install uv (default location: ~/.local/bin/uv)
curl -fsSL https://astral.sh/uv/install.sh | sh

# Ensure uv is in PATH (move binary to /usr/local/bin for non-login environments)
if [ -f "${HOME}/.local/bin/uv" ]; then
  install -m 0755 "${HOME}/.local/bin/uv" /usr/local/bin/uv
fi

# Install the requested Python version via uv
uv python install "${PYTHON_INSTALL_VERSION}"

# Find the path to that Python version
PY_BIN="$(uv python find "${PYTHON_INSTALL_VERSION}")"

# Create a virtual environment in /opt/venv
uv venv --python "${PY_BIN}" --system-site-packages /opt/venv

# Activate the venv
. /opt/venv/bin/activate

# Checks
which python
python --version

# Upgrade pip and base utilities
uv pip install --upgrade --index-url "${PIP_INDEX_URL}" pip pkginfo

which pip || true
pip --version || true

# Install core dependencies
uv pip install --no-binary :all: psutil
uv pip install --upgrade \
  setuptools \
  packaging \
  Cython \
  wheel \
  uv

# Install publishing tool
uv pip install --upgrade --index-url "${PIP_INDEX_URL}" twine

# Cleanup
rm -rf /var/lib/apt/lists/*
apt-get clean

# Symlinks for convenience
ln -sf /opt/venv/bin/python /usr/local/bin/python3
# ln -sf /opt/venv/bin/pip /usr/local/bin/pip3  # optional pip3 alias

# Final versions
which python3
python3 --version

which pip3 || true
pip3 --version || true

# Check if GIL is disabled for free-threaded builds
if [ "${PYTHON_FREE_THREADING}" = "1" ]; then
  echo ""
  echo "========================================"
  echo "🔍 VERIFYING NO-GIL PYTHON BUILD"
  echo "========================================"
  python3 -c "import sys; gil_disabled = not sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else 'N/A'; print(f'GIL Status: {\"✓ DISABLED (Free-threaded)\" if gil_disabled is True else \"✗ ENABLED\" if gil_disabled is False else \"Unknown (sys._is_gil_enabled not available)\"}')" || true
  echo "========================================"
fi
