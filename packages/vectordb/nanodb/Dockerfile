#---
# name: nanodb
# group: vectordb
# docs: docs.md
# depends: [faiss_lite, transformers, torch2trt]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/nanodb/nanodb
ENV PYTHONPATH=${PYTHONPATH}:/opt/nanodb

COPY requirements.txt *.py ./

# cannot uninstall 'blinker': It is a distutils installed project
RUN pip3 install --ignore-installed --no-cache-dir blinker && \
    pip3 install --no-cache-dir --verbose -r requirements.txt

RUN PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2` && \
    sed 's|if utils\.validate_url(img)|if False|g' -i $PYTHON_ROOT/gradio/components.py && \
    sed 's|file_path = self.make_temp_copy_if_needed(img)|file_path=img|g' -i $PYTHON_ROOT/gradio/components.py && \
    grep -B 35 'Cannot process type as image' $PYTHON_ROOT/gradio/components.py

WORKDIR /
