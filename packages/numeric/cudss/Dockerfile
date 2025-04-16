#---
# name: cudss
# group: numerical
# depends: [cuda]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDSS_URL \
    TMP=/tmp/cudss

RUN FILENAME=$(basename $CUDSS_URL) ; \
    mkdir $TMP && cd $TMP && \
    wget --quiet --show-progress --progress=bar:force:noscroll ${CUDSS_URL} && \
    dpkg -i $FILENAME && \
    cp /var/cudss-local-*/cudss-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y --no-install-recommends cudss && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    rm -rf $TMP
