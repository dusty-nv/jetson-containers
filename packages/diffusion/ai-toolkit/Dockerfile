# ---
# name: ai-toolkit
# group: flux
# requires: '>=35'
# depends: [pytorch, torchvision, transformers, bitsandbytes, huggingface_hub]
# notes: ai-toolkit for Jetson - https://github.com/ostris/ai-toolkit
# ---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    git submodule update --init --recursive

WORKDIR /opt/ai-toolkit

RUN python -m pip install -r requirements.txt
RUN pip3 install --force-reinstall 'numpy<2'

CMD ["/bin/bash"]
    