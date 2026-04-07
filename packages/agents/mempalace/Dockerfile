#---
# name: mempalace
# group: agents
# depends: [cudastack:standard, python]
# config: config.py
# test: test.sh
# docs: https://github.com/milla-jovovich/mempalace
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MEMPALACE_VERSION

COPY build.sh /tmp/mempalace/
RUN /tmp/mempalace/build.sh

ENV MEMPALACE_HOME=/data/mempalace
RUN mkdir -p $MEMPALACE_HOME

WORKDIR /data/mempalace
CMD ["mempalace", "status"]
