#!/usr/bin/env bash
printf "Installing pybind11[global]\n"
set -ex
apt-get update

# Install pybind11 from both apt and pip, but keep the
# one from pip as it is newer and have apt seem installed
apt-get install -y --no-install-recommends pybind11-dev
rm -rf /var/lib/apt/lists/*
apt-get clean

PYBIND_PKG_CONFIG="/usr/share/pkgconfig/pybind11.pc"
PYBIND_SHARE_CMAKE="/usr/lib/cmake/pybind11"
PYBIND_INCLUDE_DIR="/usr/include/pybind11"

rm $PYBIND_PKG_CONFIG
rm $PYBIND_SHARE_CMAKE/*.cmake
rm -rf $PYBIND_INCLUDE_DIR

pip3 install --upgrade pybind11[global]

PYTHON_ROOT="$(pip3 show pybind11 | grep Location: | cut -d' ' -f2)"
PYBIND_ROOT="$PYTHON_ROOT/pybind11"

cp $PYBIND_ROOT/share/pkgconfig/pybind11.pc $PYBIND_PKG_CONFIG
cp $PYBIND_ROOT/share/cmake/pybind11/* $PYBIND_SHARE_CMAKE/
cp -r $PYBIND_ROOT/include/pybind11 $PYBIND_INCLUDE_DIR