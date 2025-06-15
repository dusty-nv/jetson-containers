#!/usr/bin/env bash
set -ex

echo "Building videollama ${VIDEOLLAMA_VERSION}"

git clone --depth=1 --branch=v${VIDEOLLAMA_VERSION} https://github.com/DAMO-NLP-SG/VideoLLaMA3 /opt/videollama ||
git clone --depth=1 https://github.com/DAMO-NLP-SG/VideoLLaMA3 /opt/videollama

cd /opt/videollama

sed -i '/--extra-index-url https:\/\/download.pytorch.org\/whl\/cu118/d; s/==/>=/g' requirements.txt
pip3 install -U -r requirements.txt
sed -i 's/==/>=/g' pyproject.toml

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
pip3 wheel --wheel-dir=/opt/videollama/wheels/ --verbose .

ls /opt/videollama/wheels/
cd /

pip3 install /opt/videollama/wheels/videollama*.whl

twine upload --verbose /opt/videollama/wheels/videollama*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
