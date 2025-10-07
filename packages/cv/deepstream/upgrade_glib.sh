#!/usr/bin/env bash

# Migrate glib to newer version (2.76.6), as suggested in Deepstream Installation guide:
#   https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html#migrate-glib-to-newer-version

set -x

currentGlibVersion=$(pkg-config --modversion glib-2.0)
dpkg --compare-versions "$currentGlibVersion" "lt" "2.76.6"
if [[ $? == 0 ]]; then
  echo "Upgrade the current version of glib-2.0 to 2.76.6"
  uv pip install meson
  uv pip install ninja
  git clone https://github.com/GNOME/glib.git /opt/glib
  cd /opt/glib
  git checkout 2.76.6
  meson build --prefix=/usr
  ninja -C build/
  cd build/
  ninja install
  pkg-config --modversion glib-2.0
else
  echo "Current version of glib-2.0 is >= 2.76.6.  No upgrade needed"
fi
