#!/usr/bin/env bash
# Python installer via uv
set -euxo pipefail

# Expected variables:
#   PYTHON_VERSION (e.g., 3.12)
#   PIP_INDEX_URL optional (defaults to https://pypi.org/simple)
: "${PYTHON_VERSION:?You must define PYTHON_VERSION, e.g., 3.12}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

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
uv python install "${PYTHON_VERSION}"

# Find the path to that Python version
PY_BIN="$(uv python find "${PYTHON_VERSION}")"

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
uv pip --version || true

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
