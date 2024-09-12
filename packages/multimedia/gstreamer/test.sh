#!/usr/bin/env bash
set -e

echo "testing gstreamer..."

gst-inspect-1.0 --version
gst-inspect-1.0
gst-inspect-1.0 nvvideo4linux2
gst-inspect-1.0 nvarguscamerasrc

echo "gstreamer OK"
