#---
# name: arrow
# group: core
# config: config.py
# depends: [cuda, cmake, python]
# test: test.py
# notes: installed under `/usr/local`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ARROW_BRANCH

RUN git clone --branch=${ARROW_BRANCH} --depth=1 --recursive https://github.com/apache/arrow /opt/arrow
    
RUN cd /opt/arrow/cpp && \
    mkdir build && \
    cd build && \
    cmake \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
	 -DARROW_CUDA=ON \
	 -DARROW_PYTHON=ON \
	 -DARROW_COMPUTE=ON \
	 -DARROW_CSV=ON \
	 -DARROW_DATASET=ON \
	 -DARROW_FILESYSTEM=ON \
	 -DARROW_HDFS=ON \
	 -DARROW_JSON=ON \
	 -DARROW_PARQUET=ON \
	 -DARROW_ORC=ON \
	 -DARROW_WITH_BZ2=ON \
	 -DARROW_WITH_LZ4=ON \
	 -DARROW_WITH_SNAPPY=ON \
	 -DARROW_WITH_ZLIB=ON \
	 -DARROW_WITH_ZSTD=ON \
	 ../ && \
    make -j$(nproc) && \
    make install
    
#RUN pip3 install --no-cache-dir --verbose -r /opt/arrow/python/requirements-wheel-build.txt

RUN export PYARROW_WITH_CUDA=1 && \
    export PYARROW_WITH_HDFS=1 && \
    export PYARROW_WITH_DATASET=1 && \
    export PYARROW_WITH_PARQUET=1 && \
    export PYARROW_WITH_ORC=1 && \
    export PYARROW_PARALLEL=$(nproc) && \
    #export ARROW_HOME=/usr/local && \
    #export PYARROW_CMAKE_OPTIONS="-DARROW_HOME=/usr/local" && \
    cd /opt/arrow/python && \
    python3 setup.py bdist_wheel && \
    cp dist/*.whl /opt
    
RUN pip3 install --no-cache-dir --verbose /opt/pyarrow*.whl
RUN pip3 show pyarrow && python3 -c 'import pyarrow; print(pyarrow.__version__);'
