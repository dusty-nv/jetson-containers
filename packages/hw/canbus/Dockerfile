#---
# name: canbus
# group: hardware
# depends: [python]
# test: test.sh
# notes: CAN bus utilities for Linux and Python
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  can-utils \
          iproute2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN uv pip install python-can[serial]
