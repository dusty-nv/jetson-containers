#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${MOONCAKE_VERSION} --depth=1 --recursive https://github.com/kvcache-ai/mooncake /opt/mooncake || \
git clone --depth=1 --recursive https://github.com/kvcache-ai/Mooncake /opt/mooncake

export MAX_JOBS=$(nproc)
cd /opt/mooncake/mooncake-wheel
sed -i -e 's/2\.17/2\.35/g' setup.py

cd /opt/mooncake/scripts

echo "WARNING: AVOIDING auditwheel repair"
sed -i '/^auditwheel repair/,/^mv \${REPAIRED_DIR}\/\*\.whl \${OUTPUT_DIR}\/$/d' build_wheel.sh

cd /opt/mooncake/
bash dependencies.sh
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j
make install
ldconfig

cd /opt/mooncake/
bash ./scripts/build_wheel.sh
ls /opt/mooncake/mooncake-wheel/dist
uv pip install /opt/mooncake/mooncake-wheel/dist/mooncake_transfer_engine*.whl

cd /opt/mooncake

# Optionally upload to a repository using Twine
twine upload --verbose /opt/mooncake/mooncake-wheel/dist/mooncake_transfer_engine*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
