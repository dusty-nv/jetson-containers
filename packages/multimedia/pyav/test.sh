#!/usr/bin/env bash
echo "testing PyAV..."
set -ex

ffmpeg -version
ffmpeg -decoders
ffmpeg -decoders | grep av1

uv pip show av
python3 -c 'import av; print("pyav version:", av.__version__)'
