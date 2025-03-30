#!/usr/bin/env bash
set -ex

# if you want optimum[exporters,onnxruntime] see the optimum package
pip3 install accelerate
pip3 install sentencepiece
pip3 install optimum

# Check if TRANSFORMERS_PACKAGE is set and install it
if [ -n "$TRANSFORMERS_PACKAGE" ]; then
    pip3 install "$TRANSFORMERS_PACKAGE"
else
    echo "TRANSFORMERS_PACKAGE is not set. Skipping transformers installation."
fi

# Correct the bitsandbytes installation
pip3 install --force-reinstall scipy bitsandbytes==0.38.1

# "/usr/local/lib/python3.8/dist-packages/transformers/modeling_utils.py", line 118
# AttributeError: module 'torch.distributed' has no attribute 'is_initialized'
if [ "$(lsb_release -rs)" = "20.04" ]; then
    PYTHON_ROOT=$(pip3 show transformers | grep Location: | cut -d' ' -f2)
    sed -i -e 's|torch.distributed.is_initialized|torch.distributed.is_available|g' \
            ${PYTHON_ROOT}/transformers/modeling_utils.py
fi

# Clone the bitsandbytes repository correctly
git clone --branch=main --recursive --depth=1 https://github.com/TimDettmers/bitsandbytes.git /opt/bitsandbytes
