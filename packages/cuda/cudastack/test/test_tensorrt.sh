#!/usr/bin/env bash
set -euo pipefail

/usr/src/tensorrt/bin/trtexec --help

python3 -c "import tensorrt; print('TensorRT version:', tensorrt.__version__)"
