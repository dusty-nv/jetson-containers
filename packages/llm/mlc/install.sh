#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of MLC ${MLC_VERSION} (commit=${MLC_COMMIT})"
	exit 1
fi

torch_version=$(python3 -c 'import torch; print(torch.__version__)')

# install the wheels
if [[ -n "$(ls /tmp/mlc/*.whl)" ]]; then
    uv pip install /tmp/mlc/*.whl
else
    uv pip install tvm==${TVM_VERSION} mlc-llm==${MLC_VERSION}
    uv pip install mlc-chat==${MLC_VERSION} || echo "failed to pip install mlc-chat==${MLC_VERSION} (this is expected for mlc>=0.1.1)"
fi

# restore versions from the build
uv pip install 'pydantic>2' torch==$torch_version

# we need the source because the MLC model builder relies on it
git clone https://github.com/mlc-ai/mlc-llm ${SOURCE_DIR}
cd ${SOURCE_DIR}
git checkout ${MLC_COMMIT}
git submodule update --init --recursive

# apply patches to the source
if [ -s /tmp/mlc/patch.diff ]; then
	git apply /tmp/mlc/patch.diff
fi

# add extras to the source
cd /
cp /tmp/mlc/benchmark*.py ${SOURCE_DIR}/

# make the CUTLASS sources available for model builder
ln -s ${SOURCE_DIR}/3rdparty/tvm/3rdparty $(uv pip show tvm | awk '/Location:/ {print $2}')/tvm/3rdparty

# make sure it loads
uv pip show tvm mlc_llm

#python3 -m mlc_llm.build --help
#python3 -c "from mlc_chat import ChatModule; print(ChatModule)"
