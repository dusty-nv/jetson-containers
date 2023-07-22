#---
# name: cudf
# group: rapids
# config: config.py
# depends: [cmake, python, cupy, numba, protobuf:apt]
# test: [test_cudf.py, test_dask.py]
# notes: installed under /usr/local
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDF_VERSION
ARG CUDF_CMAKE_CUDA_ARCHITECTURES
ARG INSTALL_PREFIX=/usr/local

# force rebuild on new git commits - https://stackoverflow.com/a/56945508
ADD https://api.github.com/repos/dusty-nv/cudf/git/refs/heads/${CUDF_VERSION} /tmp/cudf_version.json

# build with the script for easier clean-up after
#COPY build_cudf.sh /tmp/build_cudf.sh
#RUN cd /tmp && ./build_cudf.sh ${CUDF_VERSION} ${CUDF_CMAKE_CUDA_ARCHITECTURES}

# DEBUG build in dockerfile
ARG BUILD_DIR=/opt
WORKDIR ${BUILD_DIR}

#
# cudf bundles many of it's dependencies, but some are still needed 
# libssl for cudf, boost and liblz4 for ORC extensions
#
RUN apt-get update && \    
    apt-get install -y --no-install-recommends \
		  libssl-dev \
		  libboost-system-dev \
		  libboost-filesystem-dev \
		  liblz4-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
# arrow gets confused if python 3.9 is present 
RUN apt-get purge -y python3.9 libpython3.9* || echo "python3.9 not found, skipping removal"

# cudf.DataFrame.sort_values() - ValueError: Cannot convert value of type NotImplementedType to cudf scalar
# https://stackoverflow.com/questions/73928178/cannot-convert-value-of-type-notimplementedtype-to-cudf-scalar-appearing-on-tr
RUN pip3 install --no-cache-dir --verbose 'numpy<1.23'


# 
# build libcudf (C++)
#
RUN git clone --branch ${CUDF_VERSION} --depth=1 https://github.com/dusty-nv/cudf && \ 
    cd cudf && \ 
    ./build.sh libcudf -v --cmake-args=\"-DCUDF_ENABLE_ARROW_S3=OFF -DCUDF_ENABLE_ARROW_PYTHON=ON -DCUDF_ENABLE_ARROW_PARQUET=ON -DCUDF_ENABLE_ARROW_ORC=ON\"


#
# build rmm
#
RUN cd cudf/cpp/build/_deps/rmm-src/python && \ 
    sed -i "s|versioneer.get_version()|\"${CUDF_VERSION}\".lstrip('v')|g" setup.py && \ 
    sed -i "s|get_versions().*|\"${CUDF_VERSION}\".lstrip('v')|g" rmm/__init__.py && \ 
    python3 setup.py bdist_wheel --verbose && \ 
    cp dist/rmm*.whl /opt && \ 
    pip3 install --no-cache-dir --verbose /opt/rmm*.whl

RUN pip3 show rmm && python3 -c 'import rmm; print(rmm.__version__)'


# 
# build pyarrow
#
RUN export PYARROW_WITH_ORC=1 && \
    export PYARROW_WITH_CUDA=1 && \
    export PYARROW_WITH_HDFS=1 && \
    export PYARROW_WITH_DATASET=1 && \
    export PYARROW_WITH_PARQUET=1 && \
    export PYARROW_PARALLEL=$(nproc) && \
    export PYARROW_CMAKE_OPTIONS="-DARROW_HOME=/usr/local" && \
    cd cudf/cpp/build/_deps/arrow-src/python && \
    python3 setup.py --verbose build_ext --inplace bdist_wheel && \
    cp dist/pyarrow*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/pyarrow*.whl

RUN pip3 show pyarrow && python3 -c 'import pyarrow; print(pyarrow.__version__)'


#
# build cudf (python)
#
RUN cd cudf/python/cudf && \
    sed -i "s|versioneer.get_version()|\"${CUDF_VERSION}\".lstrip('v')|g" setup.py && \
    sed -i "s|get_versions().*|\"${CUDF_VERSION}\".lstrip('v')|g" cudf/__init__.py && \
    PARALLEL_LEVEL=$(nproc) python3 setup.py --verbose build_ext --inplace -j$(nproc) bdist_wheel && \
    cp dist/cudf*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/cudf*.whl
    
# cudf/utils/metadata/orc_column_statistics_pb2.py
# TypeError: Descriptors cannot not be created directly.
# If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0
RUN pip3 install --no-cache-dir --verbose 'protobuf<3.20'

# make sure cudf loads
RUN pip3 show cudf && python3 -c 'import cudf; print(cudf.__version__)'


#
# build dask_cudf
#
RUN cd $BUILD_DIR/cudf/python/dask_cudf && \
    sed -i "s|versioneer.get_version()|\"${CUDF_VERSION}\".lstrip('v')|g" setup.py && \
    sed -i "s|get_versions().*|\"${CUDF_VERSION}\".lstrip('v')|g" dask_cudf/__init__.py && \
    PARALLEL_LEVEL=$(nproc) python3 setup.py --verbose build_ext --inplace -j$(nproc) bdist_wheel && \
    cp dist/dask_cudf*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/dask_cudf*.whl 

RUN pip3 show dask_cudf && python3 -c 'import dask_cudf; print(dask_cudf.__version__)'  

WORKDIR /
