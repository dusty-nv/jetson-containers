#!/usr/bin/env bash
# install python packages required for running build.sh/autotag
# and link these scripts under /usr/local so they're in the path
set -ex

ROOT="$(dirname "$(readlink -f "$0")")"
INSTALL_PREFIX="/usr/local/bin"
LSB_RELEASE="$(lsb_release -rs)"

# use virtualenv if 24.04
if [ "$LSB_RELEASE" = "24.04" ]; then
  # ensure python3-venv is available
  if ! dpkg -s python3-venv >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3-venv
  fi

  VENV="$ROOT/venv"
  mkdir -p "$VENV" || echo "warning:  $VENV either previously existed, or failed to be created"
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
fi

# install pip if needed
pip3 --version || sudo apt-get install -y python3-pip

# install package requirements
pip3 install -r "$ROOT/requirements.txt"

# link scripts to path
sudo ln -sf "$ROOT/autotag" "$INSTALL_PREFIX/autotag"
sudo ln -sf "$ROOT/jetson-containers" "$INSTALL_PREFIX/jetson-containers"
