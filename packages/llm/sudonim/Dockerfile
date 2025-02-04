#---
# name: sudonim
# group: llm
# depends: [python]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/dusty-nv/sudonim/git/refs/heads/main /tmp/sudonim_version.json

RUN git clone https://github.com/dusty-nv/sudonim /opt/sudonim && \
    pip3 install -e /opt/sudonim
  