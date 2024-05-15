#!/usr/bin/env bash
# eudev
echo "testing eudev..."

dpkg -l | grep udev

which udevd

udev --version
udevd --version

echo "eudev OK"
