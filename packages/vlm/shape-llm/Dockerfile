#---
# name: shape-llm
# alias: shape_llm
# group: vlm
# depends: [transformers, bitsandbytes, flash-attention, opencv]
# requires: '>=35'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/qizekun/ShapeLLM/git/refs/heads/main /tmp/shape_llm_version.json

RUN git clone --branch=main --depth=1 --recursive https://github.com/qizekun/ShapeLLM /opt/ShapeLLM && \
    cd /opt/ShapeLLM && \
    sed -i 's|opencv-python==.*"|opencv-python"|' pyproject.toml && \
    sed -i 's/==/>=/g' pyproject.toml && \
    sed -i 's/~=/>=/g' pyproject.toml && \
    pip3 install -e '.[all]'

#RUN pip3 install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch /opt/pointnet2 && \
    cd /opt/pointnet2 && \
    pip3 install -e pointnet2_ops_lib
