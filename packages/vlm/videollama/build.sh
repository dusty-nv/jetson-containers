#!/usr/bin/env bash
set -ex

echo "Building videollama ${VIDEOLLAMA_VERSION}"

git clone --depth=1 --branch=v${VIDEOLLAMA_VERSION} https://github.com/DAMO-NLP-SG/VideoLLaMA3 /opt/videollama ||
git clone --depth=1 https://github.com/DAMO-NLP-SG/VideoLLaMA3 /opt/videollama

cd /opt/videollama

sed -i -e 's/^decord==0\.6\.0$/decord2>=1.0.0/' requirements.txt
sed -i -E 's/"decord==0\.6\.0"([^0-9])/\"decord2>=1.0.0\"\1/' pyproject.toml
sed -i '/--extra-index-url https:\/\/download.pytorch.org\/whl\/cu118/d; s/==/>=/g' requirements.txt
sed -i -e 's/^decord==0\.6\.0$/decord2>=1.0.0/' requirements.txt
uv pip install -U -r requirements.txt
sed -i 's/==/>=/g' pyproject.toml
sed -i 's/~=/>=/g' pyproject.toml

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

MAX_JOBS="$(nproc)" \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
uv build --wheel --no-deps --out-dir /opt/videollama/wheels/ --verbose .

ls /opt/videollama/wheels/
cd /

uv pip install /opt/videollama/wheels/videollama*.whl

uv pip install --force-reinstall "transformers<=4.52"

twine upload --verbose /opt/videollama/wheels/videollama*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
