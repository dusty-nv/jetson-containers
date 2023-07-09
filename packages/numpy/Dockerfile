#
# name: numpy
# depends: [build_essential, python]
# description: install/upgrade numpy
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  python3-numpy \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
RUN pip3 show numpy && python3 -c 'import numpy; print(numpy.__version__)'
RUN pip3 install --upgrade --no-cache-dir --verbose numpy
RUN pip3 show numpy && python3 -c 'import numpy; print(numpy.__version__)'
