#---
# name: mujoco
# group: sim
# depends: [numpy]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libglfw*-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && pip3 install --no-cache-dir --verbose mujoco

ENV MUJOCO_GL=egl