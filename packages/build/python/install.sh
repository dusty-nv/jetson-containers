#!/usr/bin/env bash
###############################################################################
# Robust Python installer for jetson‑containers
#   1) Try Ubuntu official repos
#   2) Fall back to Deadsnakes PPA
#   3) Fall back to building CPython from source
#   4) On Ubuntu 24.04 (or any Python ≥3.12) create a venv in /opt/venv
#      so that all pip installs are isolated from the base image (PEP 668)
###############################################################################
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"    # passed from Dockerfile
PREFIX_VENV="/opt/venv"                     # where the venv will live
# ---------------------------------------------------------------------------

echo "==> Installing base prerequisites…"
apt-get update -qq
apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release build-essential software-properties-common

# ---------------------------------------------------------------------------
install_from_repo() {
    echo "==> Trying Ubuntu repos for Python ${PYTHON_VERSION}…"
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev
}

install_from_deadsnakes() {
    echo "==> Trying Deadsnakes PPA for Python ${PYTHON_VERSION}…"
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev
}

build_from_source() {
    echo "==> Building Python ${PYTHON_VERSION} from source (this may take a while)…"
    TMPDIR="$(mktemp -d)"
    curl -fsSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" \
        | tar -xJ -C "${TMPDIR}" --strip-components=1
    pushd "${TMPDIR}"
    ./configure --prefix=/usr --enable-optimizations --with-lto
    make -j"$(nproc)"
    make altinstall              # installs /usr/bin/python3.X
    popd
    rm -rf "${TMPDIR}"
}
# ---------------------------------------------------------------------------

if install_from_repo; then
    echo "==> Installed from Ubuntu repos."
elif install_from_deadsnakes; then
    echo "==> Installed from Deadsnakes PPA."
else
    echo "==> Neither repo had ${PYTHON_VERSION}; building from source."
    build_from_source
fi

echo "==> Python binary: $(command -v python${PYTHON_VERSION})"
ln -sf "/usr/bin/python${PYTHON_VERSION}" /usr/local/bin/python3
python3 --version

# ---------------------------------------------------------------------------
# Decide whether we need the venv path
#   • Required on Ubuntu 24.04 (noble) because Python 3.12+ is PEP 668
#   • Optional on older releases – we create it only when needed
# ---------------------------------------------------------------------------
create_venv=false
distro=$(lsb_release -rs)   # "24.04", "22.04", …
if [[ "$distro" == "24.04" ]]; then
    create_venv=true
fi

# Extra safety: if the interpreter itself is ≥3.12, enable venv path even
# on back‑ported images
if python${PYTHON_VERSION} -c 'import sys,sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)'; then
    create_venv=true
fi

# ---------------------------------------------------------------------------
if $create_venv; then
    echo "==> Creating virtual environment at ${PREFIX_VENV} to bypass PEP 668…"
    apt-get install -y --no-install-recommends python3-venv

    # ►►  skip Ubuntu's disabled ensurepip
    python3 -m venv --system-site-packages --without-pip "${PREFIX_VENV}"
    # ►►  activate the venv for the rest of this RUN layer
    # shellcheck disable=SC1090
    source "${PREFIX_VENV}/bin/activate"

    # ►►  install pip manually
    echo "==> Installing pip *inside* the venv…"
    curl -sS https://bootstrap.pypa.io/get-pip.py | python - --no-cache-dir

    PIP_BIN="${PREFIX_VENV}/bin/pip"
    PY_BIN="${PREFIX_VENV}/bin/python"
else
    echo "==> Using system site‑packages (Ubuntu ≤22.04)…"
    # Allow global pip installs in PEP 668 env just in case
    export PIP_BREAK_SYSTEM_PACKAGES=1
    PIP_BIN="pip3"
    PY_BIN="python3"
fi

# Make sure downstream layers find pip3 even if they forget the venv path
ln -sf "${PIP_BIN}" /usr/local/bin/pip3

echo "==> Pip binary: ${PIP_BIN}"
"${PIP_BIN}" --version

# ---------------------------------------------------------------------------
echo "==> Upgrading core packaging stack…"
"${PY_BIN}" -m pip install --upgrade pip pkginfo \
    setuptools packaging wheel Cython uv twine \
    --index-url https://pypi.org/simple

echo "==> Installing psutil from source (needs C extensions)…"
"${PY_BIN}" -m pip install --no-binary :all: psutil

# ---------------------------------------------------------------------------
# Clean apt caches to keep the image slim
rm -rf /var/lib/apt/lists/*
apt-get clean

echo "==> Finished. Active python: $(${PY_BIN} --version)"
echo "           Active pip:     $(${PIP_BIN} --version)"
