#---
# name: docker
# group: build
# depends: [python]
# docs: docs.md
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  apt-transport-https \
		  ca-certificates \
		  iptables \
		  lsb-release \
		  docker.io \
		  docker-compose \
		  docker-compose-v2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# so that we can run jetson-containers script from within containers
ADD https://raw.githubusercontent.com/dusty-nv/jetson-containers/master/requirements.txt /tmp/container_requirements.txt
RUN pip3 install --no-cache-dir --verbose -r /tmp/container_requirements.txt
