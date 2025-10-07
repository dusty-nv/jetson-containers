#!/usr/bin/env bash
set -ex

echo "Building ExLlamaV3 ${EXLLAMA_VERSION}"

cd /opt/exllamav3

#uv build --wheel --out-dir /opt --verbose .
uv pip install -U -r requirements.txt
uv pip install -U -r requirements_examples.txt

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd /
#rm -rf exllamav3
ls -ll /opt

uv pip install /opt/exllamav3*.whl
uv pip show exllamav3 && python3 -c 'import exllamav3'

twine upload --verbose /opt/exllamav3*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
