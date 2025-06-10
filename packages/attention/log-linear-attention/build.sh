#!/usr/bin/env bash
set -ex

echo "Building LOG_LINEAR_ATTN_VERSION ${LOG_LINEAR_ATTN_VERSION}"

git clone --recursive --depth=1 --branch=v${LOG_LINEAR_ATTN_VERSION} https://github.com/HanGuo97/log-linear-attention /opt/log-linear-attention ||
git clone --recursive --depth=1 https://github.com/HanGuo97/log-linear-attention /opt/log-linear-attention

cd /opt/log-linear-attention
# export MAX_JOBS="$(nproc)" this breaks with actual log-linear-attention
if [[ -z "${IS_SBSA}" || "${IS_SBSA}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
    export MAX_JOBS=6
else
    export MAX_JOBS="$(nproc)"
fi
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

pip3 install -r requirements.txt
# pip3 install -e /opt/log-linear-attention/flame/
# pip3 install -r flame/3rdparty/torchtitan/requirements.txt

MAX_JOBS=$MAX_JOBS \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
pip3 wheel . -v --no-deps -w /opt/log-linear-attention/wheels/

ls /opt
cd /

pip3 install /opt/log-linear-attention/wheels/hattention*.whl
#pip3 show flash-attn && python3 -c 'import flash_attn'

twine upload --verbose /opt/log-linear-attention/wheels/hattention*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"