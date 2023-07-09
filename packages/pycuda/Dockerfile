#
# name: pycuda
# category: cuda
# depends: [build_essential, python]
# test: test.py
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

RUN pip3 install --no-cache-dir --verbose pycuda six
