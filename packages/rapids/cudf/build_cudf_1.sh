#!/usr/bin/env bash
set -e

CUDF_VERSION=${1:-"$CUDF_VERSION"}

export CUDF_CMAKE_CUDA_ARCHITECTURES=${2:-"$CUDF_CMAKE_CUDA_ARCHITECTURES"}
export INSTALL_PREFIX=/usr/local

# build under /tmp (sources are removed later)
BUILD_DIR=/tmp
cd $BUILD_DIR


cd / && pip3 show pyarrow && python3 -c 'import pyarrow; print(pyarrow.__version__)'


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
cd / && pip3 show cudf && python3 -c 'import cudf; print(cudf.__version__)'


#
# build dask_cudf
#
cd $BUILD_DIR/cudf/python/dask_cudf

sed -i "s|versioneer.get_version()|\"${CUDF_VERSION}\".lstrip('v')|g" setup.py
sed -i "s|get_versions().*|\"${CUDF_VERSION}\".lstrip('v')|g" dask_cudf/__init__.py

PARALLEL_LEVEL=$(nproc) python3 setup.py --verbose build_ext --inplace -j$(nproc) bdist_wheel

cp dist/dask_cudf*.whl /opt
pip3 install --no-cache-dir --verbose /opt/dask_cudf*.whl 

cd / && pip3 show dask_cudf && python3 -c 'import dask_cudf; print(dask_cudf.__version__)'  


#
# cleanup
#
rm -rf $BUILD_DIR/cudf 
rm -rf /var/lib/apt/lists/*

apt-get clean
