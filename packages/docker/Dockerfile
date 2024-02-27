#---
# name: docker
# group: core
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
