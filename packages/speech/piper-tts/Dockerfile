#---
# name: piper-tts
# group: audio
# depends: [onnxruntime]
# test: [test_piper_phonemize.sh, test.py]
#---
    ARG BASE_IMAGE
    FROM ${BASE_IMAGE}
    
    # Install required packages
    RUN apt-get update && apt-get install -y \
        git \
        build-essential \
        cmake \
        wget \
        tar \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*
    
    # Add this after the apt-get install section:
    RUN pip3 install --no-cache-dir "numpy<2.0"
    
    # Set onnxruntime version and architecture (for aarch64 on Jetson)
    ARG ONNXRUNTIME_VERSION=1.14.0
    ARG ONNXRUNTIME_ARCH=aarch64
    
    # Download and install the onnxruntime C++ development package for ARM
    RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz && \
        tar -xzvf onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz && \
        cp -r onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}/include/* /usr/local/include/ && \
        cp -r onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}/lib/* /usr/local/lib/ && \
        rm -rf onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION} onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz
    
    # Ensure the dynamic linker finds the onnxruntime library
    ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    
    ARG PIPER_VERSION=master \
        PIPER_PHONEMIZE_VERSION=master \
        ONNXRUNTIME_DIR=/usr/local
    
    # make PIPER_CACHE a default mounted location available for downloading the models
    # this still needs to be explicitly passed to piper.get_voices(), ensure_path_exists(), ect.
    ENV PIPER_CACHE=/data/models/piper \
        PIPER_PHONEMIZE_DIR=/usr/local \
        ESPEAK_NG_DATA_DIR=/usr/local/share/espeak-ng-data
    
    #ENV LD_LIBRARY_PATH=/usr/local/:$LD_LIBRARY_PATH
    
    WORKDIR /opt
    
    # Even though piper will download a piper-phonemize version
    # automatically, we cannot use it because it will not use
    # our GPU accelerated ONNXRUNTIME. We need to manually build
    # piper-phonemize to  pass the appropriate flags and have it
    # use our ORT (which is already in /usr/local/).
    RUN git clone --branch ${PIPER_PHONEMIZE_VERSION} --depth 1 https://github.com/rhasspy/piper-phonemize && \
        cd piper-phonemize && \
        cmake -B build -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} -DCMAKE_CXX_FLAGS="-I${ONNXRUNTIME_DIR}/include" && \
        cmake --build build --config Release --parallel && \
        cmake --install build && \
        CPPFLAGS="-I${ONNXRUNTIME_DIR}/include" \
        pip3 install --no-cache-dir --verbose . && \
        ln -s ${ESPEAK_NG_DATA_DIR} /usr/share/espeak-ng-data
    
    # Now we can build piper. In order to avoid it from downloading
    # automatically piper-phonemize, we specify an installation dir.
    RUN git clone --branch ${PIPER_VERSION} --depth 1 https://github.com/rhasspy/piper && \
        cd piper && \
        cmake -B build -DPIPER_PHONEMIZE_DIR=${PIPER_PHONEMIZE_DIR} -DCMAKE_CXX_FLAGS="-I${ONNXRUNTIME_DIR}/include" && \
        cmake --build build --config Release --parallel && \
        cmake --install build && \
        cd src/python_run && \
        pip3 install --ignore-installed --no-cache-dir blinker && \
        pip3 install -r requirements_http.txt && \
        pip3 install --no-cache-dir --verbose --no-deps .[gpu,http]
    
    # Install the onnxruntime Python package to resolve the ModuleNotFoundError
    RUN pip3 install --no-cache-dir onnxruntime==${ONNXRUNTIME_VERSION}
    
    WORKDIR /
    