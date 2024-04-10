#!/usr/bin/env bash
# faiss
set -ex

echo "Building faiss ${FAISS_VERSION} (branch=${FAISS_BRANCH})"
 
# workaround for 'Could NOT find Python3 (missing: Python3_NumPy_INCLUDE_DIRS Development'
# update-alternatives --install /usr/bin/python python /usr/bin/python3 1
apt purge -y python3.9 libpython3.9* || echo "python3.9 not found, skipping removal"
ls -ll /usr/bin/python*
    
# clone sources
git clone https://github.com/facebookresearch/faiss /opt/faiss && \
cd /opt/faiss
git checkout ${FAISS_BRANCH}

# build C++
install_dir="/opt/faiss/install"

mkdir build
cd build

cmake \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_RAFT=OFF \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
  -DCMAKE_INSTALL_PREFIX=${install_dir} \
  ../
  
make -j$(nproc) faiss
make install

#make demo_ivfpq_indexing
#make demo_ivfpq_indexing_gpu 
    
# build python
make -j$(nproc) swigfaiss

cd faiss/python
python3 setup.py --verbose bdist_wheel --dist-dir /opt

pip3 install --no-cache-dir --verbose /opt/faiss*.whl
pip3 show faiss && python3 -c 'import faiss'

# cache build artifacts on server
twine upload --verbose /opt/faiss*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
tarpack upload faiss-${FAISS_VERSION} ${install_dir} || echo "failed to upload tarball"

# install local copy
cp -r ${install_dir}/* /usr/local/