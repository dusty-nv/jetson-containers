#---
# name: shape-llm
# alias: shape_llm
# group: vlm
# depends: [cmake, rust, pytorch, torchvision, transformers, bitsandbytes, flash-attention, opencv]
# requires: '>=35'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/qizekun/ShapeLLM/git/refs/heads/main /tmp/shape_llm_version.json

RUN git clone --branch=main --depth=1 --recursive https://github.com/qizekun/ShapeLLM /opt/ShapeLLM && \
    cd /opt/ShapeLLM && \
    sed -i 's/opencv-python==4.10.0.84/opencv-contrib-python/g' pyproject.toml && \
    sed -i 's/tokenizers>=0.12.1,<0.14/tokenizers>=0.21.2/g' pyproject.toml && \
    sed -i 's/==/>=/g' pyproject.toml && \
    sed -i 's/~=/>=/g' pyproject.toml && \
    uv pip install tokenizers && \
    uv pip install -e '.[all]'

#RUN uv pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch /opt/pointnet2 && \
    cd /opt/pointnet2 && \
    uv pip install -e pointnet2_ops_lib
