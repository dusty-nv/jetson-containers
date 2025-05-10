#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${MOONCAKE_VERSION} --depth=1 --recursive https://github.com/kvcache-ai/mooncake /opt/mooncake || \
git clone --depth=1 --recursive https://github.com/kvcache-ai/Mooncake /opt/mooncake

cd /opt/mooncake

export MAX_JOBS=$(nproc)

bash dependencies.sh
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j
make install
ldconfig

cd /opt/mooncake/mooncake-wheel
sed -i 's/manylinux_2_35_x86_64/manylinux_2_35_aarch64/' setup.py

cd /opt/mooncake/scripts
sed -i 's/--plat manylinux_2_35_x86_64/--plat manylinux_2_35_aarch64/' build_wheel.sh
sed -i '/^auditwheel repair/,/^mv \${REPAIRED_DIR}\/\*\.whl \${OUTPUT_DIR}\/$/d' build_wheel.sh


cd /opt/mooncake/
bash ./scripts/build_wheel.sh
ls /opt/mooncake/mooncake-wheel/dist
pip3 install /opt/mooncake/mooncake-wheel/dist/mooncake_transfer_engine*.whl

cd /opt/mooncake

# Optionally upload to a repository using Twine
twine upload --verbose /opt/mooncake/mooncake-wheel/dist/mooncake_transfer_engine*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
