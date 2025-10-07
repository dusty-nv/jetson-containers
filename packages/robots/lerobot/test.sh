#!/usr/bin/env bash
set -ex

uv pip show lerobot
python3 -c 'import lerobot; print("lerobot version:", lerobot.__version__)'
