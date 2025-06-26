#---
# name: faster-whisper
# group: audio
# depends: [ctranslate2, onnxruntime, huggingface_hub, numpy]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN git clone https://github.com/guillaumekln/faster-whisper /opt/faster-whisper && \
    cd /opt/faster-whisper && \
    sed -i 's/==/>=/g' requirements.txt && \
    sed -i 's/~=/>=/g' requirements.txt && \
    cat requirements.txt && \
    pip3 install -U -r requirements.txt && \
    python3 setup.py bdist_wheel

RUN cp /opt/faster-whisper/dist/faster_whisper*.whl /opt && \
    pip3 install /opt/faster_whisper*.whl

RUN pip3 show faster_whisper && python3 -c 'import faster_whisper'
