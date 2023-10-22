#!/usr/bin/env bash

set -e

cd /opt/stable-diffusion-webui && python3 launch.py --data=/data/models/stable-diffusion --enable-insecure-extension-access --xformers --listen --port=7860 "$@"
