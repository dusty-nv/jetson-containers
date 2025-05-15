#---
# name: pizero
# group: robots
# docs: docs.md
# depends: [jax, pytorch, transformers, lerobot, opencv:4.11.0, pyav]
# requires: '>=36'
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/pizero

ARG PIZERO_REPO=Physical-Intelligence/openpi \
    PIZERO_BRANCH=main

# Clone and build pizero
RUN git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi /opt/pizero && \
    cd /opt/pizero && \
    sed -i '/"torch[^"]*",/d' pyproject.toml && \
    sed -i '/"opencv-python[^"]*",/d' pyproject.toml && \
    sed -i '/"transformers[^"]*",/d' pyproject.toml && \
    sed -i '/"lerobot[^"]*",/d' pyproject.toml && \
    sed -i '/"jax[^"]*",/d' pyproject.toml && \
    sed -i '/"av[^"]*",/d' pyproject.toml && \
    cat -n pyproject.toml

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv pip compile pyproject.toml && \
    GIT_LFS_SKIP_SMUDGE=1 uv sync && \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

RUN pip3 install --no-cache-dir --extra-index-url https://pypi.org/simple --index-url https://pypi.org/simple 'av>=12.0.5,<13.0.0'

CMD /bin/bash