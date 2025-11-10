#!/usr/bin/env bash
set -ex
echo "Building CTranslate2 ${CTRANSLATE_VERSION} (${CTRANSLATE_BRANCH})"

# clone sources
git clone --branch=${CTRANSLATE_BRANCH} --recursive https://github.com/OpenNMT/CTranslate2.git ${CTRANSLATE_SOURCE} ||
git clone --recursive https://github.com/OpenNMT/CTranslate2.git ${CTRANSLATE_SOURCE}

mkdir -p $CTRANSLATE_SOURCE/build
cd $CTRANSLATE_SOURCE/build

install_dir="${CTRANSLATE_SOURCE}/build/install"

# build C++ libraries
cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP -DCMAKE_INSTALL_PREFIX=$install_dir

make -j$(nproc)
make install

cp -r ${install_dir}/* /usr/local/
ldconfig

# build Python packages
cd $CTRANSLATE_SOURCE/python
uv pip install -r install_requirements.txt
python3 setup.py --verbose bdist_wheel --dist-dir /opt

# install/upload wheels
uv pip install --force-reinstall /opt/ctranslate2*.whl

twine upload --verbose /opt/ctranslate2*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
tarpack upload ctranslate2-${CTRANSLATE_VERSION} ${install_dir} || echo "failed to upload tarball"
