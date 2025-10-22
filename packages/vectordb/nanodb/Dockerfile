#---
# name: nanodb
# group: vectordb
# docs: docs.md
# depends: [faiss_lite, transformers, torch2trt, clip_trt]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/dusty-nv/nanodb/git/refs/heads/main /tmp/nanodb_version.json

RUN git clone https://github.com/dusty-nv/NanoDB /opt/NanoDB && \
    cd /opt/NanoDB && \
    uv pip install --reinstall blinker && \
    uv pip install -r requirements.txt && \
    uv pip install -e .

RUN PYTHON_ROOT=`uv pip show torch | grep Location: | cut -d' ' -f2` && \
    sed 's|if utils\.validate_url(img)|if False|g' -i $PYTHON_ROOT/gradio/components.py && \
    sed 's|file_path = self.make_temp_copy_if_needed(img)|file_path=img|g' -i $PYTHON_ROOT/gradio/components.py && \
    grep -B 35 'Cannot process type as image' $PYTHON_ROOT/gradio/components.py

