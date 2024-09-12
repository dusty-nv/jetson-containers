#!/usr/bin/env bash
echo "testing PyAV..."
set -ex

$HOME/bin/ffmpeg -version
$HOME/bin/ffmpeg -decoders
$HOME/bin/ffmpeg -decoders | grep av1

pip3 show pyav
python3 -c 'import av; print("pyav version:", av.__version__)'