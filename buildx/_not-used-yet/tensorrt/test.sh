#!/usr/bin/env bash

/usr/src/tensorrt/bin/trtexec --help

python3 -c "import tensorrt; print('TensorRT version:', tensorrt.__version__)" || echo "WARNING - failed to 'import tensorrt' under Python $(python3 --version)"