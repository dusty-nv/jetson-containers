#---
# name: xtuner
# group: vlm
# depends: [transformers, bitsandbytes, flash-attention]
# requires: '>=35'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/InternLM/xtuner/git/refs/heads/main /tmp/xtuner_version.json

RUN git clone --branch=main --depth=1 --recursive https://github.com/InternLM/xtuner /opt/xtuner && \
    cd /opt/xtuner && \
    sed -i 's|bitsandbytes.*|bitsandbytes|' requirements/runtime.txt && \
    sed -i 's|mpi4py-mpich.*||' requirements/deepspeed.txt && \
    sed -i 's/==/>=/g' requirements/runtime.txtml && \
    sed -i 's/~=/>=/g' requirements/runtime.txt && \
    sed -i 's/==/>=/g' requirements/deepspeed.txt && \
    sed -i 's/~=/>=/g' requirements/deepspeed.txt && \
    uv pip install -e '.[all]'

# SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
