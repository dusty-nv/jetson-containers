#!/usr/bin/env bash
set -ex

pip3 show lerobot
python3 -c 'import lerobot; print("lerobot version:", lerobot.__version__)'