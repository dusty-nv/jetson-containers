#!/usr/bin/env bash
set -e

CUDF_VERSION=${1:-"$CUDF_VERSION"}

export CUDF_CMAKE_CUDA_ARCHITECTURES=${2:-"$CUDF_CMAKE_CUDA_ARCHITECTURES"}
export INSTALL_PREFIX=/usr/local

echo "building cudf $CUDF_VERSION  CUDF_CMAKE_CUDA_ARCHITECTURES=$CUDF_CMAKE_CUDA_ARCHITECTURES  INSTALL_PREFIX=$INSTALL_PREFIX"

# build under /tmp (sources are removed later)
BUILD_DIR=/tmp
cd $BUILD_DIR

#
# cudf bundles many of it's dependencies, but some are still needed 
# libssl for cudf, boost and liblz4 for ORC extensions
#
apt-get update   
apt-get install -y --no-install-recommends \
		libssl-dev \
		libboost-system-dev \
		libboost-filesystem-dev \
		liblz4-dev

# arrow gets confused if python 3.9 is present 
apt-get purge -y python3.9 libpython3.9* || echo "python3.9 not found, skipping removal"

# cudf.DataFrame.sort_values() - ValueError: Cannot convert value of type NotImplementedType to cudf scalar
# https://stackoverflow.com/questions/73928178/cannot-convert-value-of-type-notimplementedtype-to-cudf-scalar-appearing-on-tr
pip3 install --no-cache-dir --verbose 'numpy<1.23'


# 
# build libcudf (C++)
#
git clone --branch ${CUDF_VERSION} --depth=1 https://github.com/dusty-nv/cudf 
cd cudf
./build.sh libcudf -v --cmake-args=\"-DCUDF_ENABLE_ARROW_S3=OFF -DCUDF_ENABLE_ARROW_PYTHON=ON -DCUDF_ENABLE_ARROW_PARQUET=ON -DCUDF_ENABLE_ARROW_ORC=ON\"


#
# build rmm
#
cd cpp/build/_deps/rmm-src/python

sed -i "s|versioneer.get_version()|\"${CUDF_VERSION}\".lstrip('v')|g" setup.py
sed -i "s|get_versions().*|\"${CUDF_VERSION}\".lstrip('v')|g" rmm/__init__.py

python3 setup.py bdist_wheel --verbose
cp dist/rmm*.whl /opt
pip3 install --no-cache-dir --verbose /opt/rmm*.whl
pip3 show rmm

cd / && python3 -c 'import rmm; print(rmm.__version__)'


# 
# build pyarrow
#
export PYARROW_WITH_ORC=1
export PYARROW_WITH_CUDA=1
export PYARROW_WITH_HDFS=1
export PYARROW_WITH_DATASET=1
export PYARROW_WITH_PARQUET=1
export PYARROW_PARALLEL=$(nproc)
export PYARROW_CMAKE_OPTIONS="-DARROW_HOME=/usr/local"

cd $BUILD_DIR/cudf/cpp/build/_deps/arrow-src/python

python3 setup.py --verbose build_ext --inplace bdist_wheel
cp dist/pyarrow*.whl /opt
pip3 install --no-cache-dir --verbose /opt/pyarrow*.whl
pip3 show pyarrow 

cd / && python3 -c 'import pyarrow; print(pyarrow.__version__)'


#
# build cudf (python)
#
cd $BUILD_DIR/cudf/python/cudf

sed -i "s|versioneer.get_version()|\"${CUDF_VERSION}\".lstrip('v')|g" setup.py
sed -i "s|get_versions().*|\"${CUDF_VERSION}\".lstrip('v')|g" cudf/__init__.py

PARALLEL_LEVEL=$(nproc) python3 setup.py --verbose build_ext --inplace -j$(nproc) bdist_wheel

cp dist/cudf*.whl /opt
pip3 install --no-cache-dir --verbose /opt/cudf*.whl
    
# cudf/utils/metadata/orc_column_statistics_pb2.py
# TypeError: Descriptors cannot not be created directly.
# If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0
pip3 install --no-cache-dir --verbose 'protobuf<3.20'

# make sure cudf loads
pip3 show cudf 
cd / && python3 -c 'import cudf; print(cudf.__version__)'


#
# build dask_cudf
#
cd $BUILD_DIR/cudf/python/dask_cudf

sed -i "s|versioneer.get_version()|\"${CUDF_VERSION}\".lstrip('v')|g" setup.py
sed -i "s|get_versions().*|\"${CUDF_VERSION}\".lstrip('v')|g" dask_cudf/__init__.py

PARALLEL_LEVEL=$(nproc) python3 setup.py --verbose build_ext --inplace -j$(nproc) bdist_wheel

cp dist/dask_cudf*.whl /opt
pip3 install --no-cache-dir --verbose /opt/dask_cudf*.whl 

pip3 show dask_cudf 
cd / && python3 -c 'import dask_cudf; print(dask_cudf.__version__)'  


#
# cleanup
#
rm -rf $BUILD_DIR/cudf 
rm -rf /var/lib/apt/lists/*

apt-get clean
