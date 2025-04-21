#!/usr/bin/env bash
set -ex

pip3 install --force-reinstall "cmake$1"

cmake --version
which cmake

update-alternatives --force --install /usr/bin/cmake cmake "$(which cmake)" 100

ls -ll /usr/bin/cmake*