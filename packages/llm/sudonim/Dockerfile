#---
# name: sudonim
# group: llm
# depends: [cuda, python]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SUDONIM_SOURCE_DIR=/opt/sudonim \
    SUDONIM_PATCH_DIR=off

ADD https://api.github.com/repos/dusty-nv/sudonim/git/refs/heads/main /tmp/sudonim_version.json

RUN git clone https://github.com/dusty-nv/sudonim ${SUDONIM_SOURCE_DIR} && \
    uv pip install -e ${SUDONIM_SOURCE_DIR}

COPY patches ${SUDONIM_SOURCE_DIR}/patches/

RUN if [ ${SUDONIM_PATCH_DIR} != "off" ]; then \
        cd ${SUDONIM_SOURCE_DIR}/patches/${SUDONIM_PATCH_DIR} ; \
        for script in ./*.sh; do \
            bash $script ; \
        done \
    fi
