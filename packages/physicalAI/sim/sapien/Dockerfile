#---
# name: sapien
# group: sim
# depends: [physx, numpy]
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# https://github.com/haosulab/SAPIEN/blob/dev/docker/Dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libxrandr-dev \
		  libxinerama-dev \
		  libxcursor-dev \
		  libxi-dev \
		  spirv-* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && uv pip install mujoco

RUN uv pip install git+https://github.com/fbxiang/auditwheel.git@088b034ae497b0add794da13f8e0f3b0a3b2d651
RUN echo "#!/usr/bin/env python3" > /usr/local/bin/auditwheel && \
    echo "import re, sys" >> /usr/local/bin/auditwheel && \
    echo "from auditwheel.main import main" >> /usr/local/bin/auditwheel && \
    echo "if __name__ == '__main__':" >> /usr/local/bin/auditwheel && \
    echo "    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])" >> /usr/local/bin/auditwheel && \
    echo "    sys.exit(main())" >> /usr/local/bin/auditwheel

RUN git clone --branch=dev --depth=1 --recursive https://github.com/haosulab/SAPIEN /opt/SAPIEN

COPY patches/*.diff /opt/SAPIEN/patches/
RUN cd /opt/SAPIEN && \
    git apply patches/*.diff && \
    git diff && git status

#RUN cd /opt/SAPIEN && \
#    CMAKE_BUILD_PARALLEL_LEVEL=-j$(nproc) ./build.sh $(echo "$PYTHON_VERSION" | sed 's|\.||g')
    #CXXFLAGS=-isystem\ /usr/local/cuda/include \
RUN CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    CUDA_PATH=/usr/local/cuda \
    uv build --wheel --out-dir $PIP_WHEEL_DIR --verbose /opt/SAPIEN
