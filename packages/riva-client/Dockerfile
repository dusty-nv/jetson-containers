#---
# name: riva-client
# group: ml
# depends: [python, bazel]
# requires: '>=34.1.0'
# test: [test.sh, test.py]
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


WORKDIR /opt/riva


# install prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libasound2-dev \
		  libopus0 \
		  libopus-dev \
		  libopusfile0 \
		  libopusfile-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
    
# Riva C++ client (binaries in bazel-bin/riva/clients)
ADD https://api.github.com/repos/nvidia-riva/cpp-clients/git/refs/heads/main /tmp/riva_cpp_version.json

RUN git clone --depth=1 --recursive https://github.com/nvidia-riva/cpp-clients && \
    cd cpp-clients && \
    bazel build ...
    

# Riva Python client
ADD https://api.github.com/repos/nvidia-riva/python-clients/git/refs/heads/main /tmp/riva_python_version.json

RUN git clone --depth=1 --recursive https://github.com/nvidia-riva/python-clients && \
    cd python-clients && \
    pip3 install --no-cache-dir --verbose -r requirements.txt && \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/nvidia_riva_client*.whl /opt
    
RUN pip3 install --no-cache-dir --verbose /opt/nvidia_riva_client*.whl
RUN pip3 show nvidia-riva-client && python3 -c 'import riva.client; print(riva.client.__version__)'

WORKDIR /