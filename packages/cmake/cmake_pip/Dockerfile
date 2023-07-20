#---
# name: cmake:pip
# alias: cmake
# category: base
# depends: [build-essential, python]
# notes: upgrade cmake with pip
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

    
# prior issue:  cmake was pinned to 3.22.3 because of https://github.com/pytorch/pytorch/issues/74955	 
RUN pip3 install --upgrade --force-reinstall --no-cache-dir --verbose cmake
RUN cmake --version && which cmake
