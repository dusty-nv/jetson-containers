#---
# name: audiocraft
# group: audio
# depends: [opencv, python, pytorch:2.1-distributed, torchaudio, huggingface_hub, transformers, xformers, protobuf:cpp, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

#RUN pip3 install matplotlib pycocotools tqdm
RUN pip3 install torchmetrics

# Clone the repository:
RUN git clone https://github.com/facebookresearch/audiocraft/ && \
    cd audiocraft && \
    sed 's|^torch.*||' -i requirements.txt && \
    sed 's|^huggingface_hub.*||' -i requirements.txt && \
    sed 's|^transformers.*||' -i requirements.txt && \
    sed 's|^xformers.*||' -i requirements.txt && \
    sed 's|^protobuf.*||' -i requirements.txt && \
    cat requirements.txt && \
    pip3 install .
    
RUN cd /opt/pytorch && \    
    ls -lh && \
    find ./ -name *.whl &&\
    pip3 install --verbose ./dist/*.whl

RUN python3 -c 'import torchaudio; print(torchaudio.__version__);'

WORKDIR /opt/audiocraft/
