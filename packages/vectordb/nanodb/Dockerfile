#---
# name: nanodb
# group: vectordb
# depends: [pytorch, transformers, faiss:lite, torch2trt]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/nanodb/nanodb

COPY requirements.txt .
RUN pip3 install --no-cache-dir --verbose -r requirements.txt

RUN PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2` && \
    sed 's|if utils\.validate_url(img)|if False|g' -i $PYTHON_ROOT/gradio/components.py && \
    sed 's|file_path = self.make_temp_copy_if_needed(img)|file_path=img|g' -i $PYTHON_ROOT/gradio/components.py && \
    grep -B 35 'Cannot process type as image' $PYTHON_ROOT/gradio/components.py

COPY *.py ./

ENV PYTHONPATH=${PYTHONPATH}:/opt/nanodb

WORKDIR /
