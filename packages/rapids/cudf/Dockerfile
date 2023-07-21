#---
# name: cudf
# category: rapids
# config: config.py
# depends: [cmake, python, cupy, numba, protobuf:apt]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDF_VERSION
ARG CUDA_ARCHITECTURES

# arrow gets confused if python 3.9 is present
RUN apt-get update && \    
    apt purge -y python3.9 libpython3.9* || echo "python3.9 not found, skipping removal" && \
    apt-get install -y --no-install-recommends \
		  libssl-dev \
		  libboost-system-dev \
		  libboost-filesystem-dev \
		  liblz4-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
# force rebuild on new git commits - https://stackoverflow.com/a/56945508
ADD https://api.github.com/repos/dusty-nv/cudf/git/refs/heads/${CUDF_VERSION} /tmp/cudf_version.json

# build libcudf (C++)
RUN cd /opt && \
    git clone --branch ${CUDF_VERSION} --depth=1 https://github.com/dusty-nv/cudf && \
    cd cudf && \
    ./build.sh --help && \
    export INSTALL_PREFIX=/usr/local && \
    export CUDF_CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ./build.sh libcudf -v --cmake-args=\"-DCUDF_ENABLE_ARROW_S3=OFF -DCUDF_ENABLE_ARROW_PYTHON=ON -DCUDF_ENABLE_ARROW_PARQUET=ON -DCUDF_ENABLE_ARROW_ORC=ON\"

#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
#    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN cd /opt/cudf && \
    export INSTALL_PREFIX=/usr/local && \
    export CUDF_CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    cd cpp/build/_deps/rmm-src/python && \
    python3 setup.py bdist_wheel --verbose && \
    cp dist/rmm*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/rmm*.whl

RUN python3 -c 'import rmm'

#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#		  protobuf-compiler \
#		  libprotoc-dev \
#		  libprotobuf-dev \
#    && rm -rf /var/lib/apt/lists/* \
#    && apt-get clean
    
RUN cd /opt/cudf/cpp/build/_deps/arrow-src/python && \
    export PYARROW_WITH_ORC=1 && \
    export PYARROW_WITH_CUDA=1 && \
    export PYARROW_WITH_HDFS=1 && \
    export PYARROW_WITH_DATASET=1 && \
    export PYARROW_WITH_PARQUET=1 && \
    export PYARROW_PARALLEL=$(nproc) && \
    export PYARROW_CMAKE_OPTIONS="-DARROW_HOME=/usr/local" && \
    python3 setup.py --verbose build_ext --inplace bdist_wheel && \
    cp dist/pyarrow*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/pyarrow*.whl

RUN pip3 show pyarrow && python3 -c 'import pyarrow; print(pyarrow.__version__)' 
    
    #ARROW_VERSION=$(awk -F"\"" '/ARROW_VERSION_STRING/{print $2}' /usr/local/include/arrow/util/config.h) && \
    #pip3 install --no-cache-dir --verbose pyarrow==${ARROW_VERSION} && \
    
RUN cd /opt/cudf/python/cudf && \
    export INSTALL_PREFIX=/usr/local && \
    export CUDF_CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    PARALLEL_LEVEL=$(nproc) python3 setup.py --verbose build_ext --inplace -j$(nproc) bdist_wheel
    
    #./build.sh cudf -v
    
#RUN cd /opt/cudf/python/cudf && \
#    export CUDF_CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
#    pip3 install --no-cache-dir --verbose pyarrow && \
#    python3 setup.py bdist_wheel
    
RUN cd /opt/cudf/python/cudf && \
    cp dist/cudf*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/cudf*.whl
    
# cudf/utils/metadata/orc_column_statistics_pb2.py
# TypeError: Descriptors cannot not be created directly.
# If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0
RUN pip3 install --no-cache-dir --verbose 'protobuf<3.20'

# make sure it loads
RUN pip3 show cudf && python3 -c 'import cudf; print(cudf.__version__)'

