#!/usr/bin/env bash
# eudev
set -exo pipefail

echo "testing eudev..."

dpkg -l | grep udev

which udevd

udevd --version

echo "eudev OK"
