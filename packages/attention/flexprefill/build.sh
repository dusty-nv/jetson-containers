#!/usr/bin/env bash
set -ex

echo "Building FlexPrefill ${FLEXPREFILL_VERSION}"

git clone --depth=1 --branch=v${FLEXPREFILL_VERSION} https://github.com/ByteDance-Seed/FlexPrefill /opt/flexprefill ||
git clone --depth=1 https://github.com/ByteDance-Seed/FlexPrefill /opt/flexprefill

cd /opt/flexprefill

sed -i 's/==/>=/g' extra_requirements.txt
sed -i 's/transformers==/transformers>=/; s/triton==/triton>=/' setup.py
uv pip install packaging
uv pip install --reinstall blinker


export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
python3 setup.py --verbose bdist_wheel --dist-dir /opt/flexprefill/wheels

ls /opt/flexprefill/wheels
cd /

uv pip install /opt/flexprefill/wheels/flex_prefill*.whl

twine upload --verbose /opt/flexprefill/wheels/flex_prefill*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
