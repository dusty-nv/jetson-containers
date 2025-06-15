#!/usr/bin/env bash
set -ex

# if you want optimum[exporters,onnxruntime] see the optimum package
pip3 install accelerate
pip3 install sentencepiece
pip3 install optimum

# Get the version that optimum installed, being more precise with the grep
OPTIMUM_TRANSFORMERS_VERSION=$(pip3 show transformers | grep '^Version:' | head -n1 | cut -d' ' -f2)

# install from pypi, git, ect (sometimes other version got installed)
pip3 uninstall -y transformers

# Use the version from optimum if environment variables aren't set
if [ -z "$TRANSFORMERS_VERSION" ]; then
    TRANSFORMERS_VERSION="$OPTIMUM_TRANSFORMERS_VERSION"
fi

if [ -z "$TRANSFORMERS_PACKAGE" ]; then
    TRANSFORMERS_PACKAGE="transformers==$TRANSFORMERS_VERSION"
fi

echo "Installing transformers $TRANSFORMERS_VERSION (from $TRANSFORMERS_PACKAGE)"
pip3 install ${TRANSFORMERS_PACKAGE}

# "/usr/local/lib/python3.8/dist-packages/transformers/modeling_utils.py", line 118
# AttributeError: module 'torch.distributed' has no attribute 'is_initialized'
if [ $(lsb_release -rs) = "20.04" ]; then
	PYTHON_ROOT=`pip3 show transformers | grep Location: | cut -d' ' -f2`
	sed -i -e 's|torch.distributed.is_initialized|torch.distributed.is_available|g' \
			${PYTHON_ROOT}/transformers/modeling_utils.py
fi