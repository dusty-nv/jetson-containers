#!/usr/bin/env bash
set -ex

# install the wheels (TODO specify commit)
pip3 install --no-cache-dir --verbose \
	tvm \
	mlc-llm \
	mlc-chat \
	'pydantic>2'

# we need the source because the MLC model builder relies on it
git clone https://github.com/mlc-ai/mlc-llm /opt/mlc-llm
cd /opt/mlc-llm
git checkout ${MLC_COMMIT}
git submodule update --init --recursive
    
# apply patches to the source
if [ -s /tmp/mlc/patch.diff ]; then 
	git apply /tmp/mlc/patch.diff 
fi

# add extras to the source
cd /
cp /tmp/mlc/benchmark.py /opt/mlc-llm/

# make the CUTLASS sources available for model builder
ln -s /opt/mlc-llm/3rdparty/tvm/3rdparty /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tvm/3rdparty

# make sure it loads
pip3 show mlc_llm
python3 -m mlc_llm.build --help
python3 -c "from mlc_chat import ChatModule; print(ChatModule)"
