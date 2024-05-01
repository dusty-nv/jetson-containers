#!/usr/bin/env bash
# install python packages required for running build.sh/autotag
# and link these scripts under /usr/local so they're in the path
set -ex

ROOT="$(dirname "$(readlink -f "$0")")"
INSTALL_PREFIX="/usr/local/bin"

# install pip if needed
pip3 --version || sudo apt-get install python3-pip

# install package requirements
pip3 install -r $ROOT/requirements.txt

# link scripts to path
sudo ln -sf $ROOT/autotag $INSTALL_PREFIX/autotag
sudo ln -sf $ROOT/jetson-containers $INSTALL_PREFIX/jetson-containers
