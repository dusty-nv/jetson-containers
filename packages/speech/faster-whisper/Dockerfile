#---
# name: faster-whisper
# group: audio
# depends: [ctranslate2, onnxruntime, huggingface_hub, numpy]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN git clone https://github.com/guillaumekln/faster-whisper && \
    cd faster-whisper && \
    sed -i \
        -e 's|^onnxruntime.*||g' \
        -e 's|^ctranslate2.*||g' \
        -e 's|^huggingface_hub.*||g' \
        -e 's|^tokenizers.*|tokenizers|g' \
        requirements.txt && \
    cat requirements.txt && \
    pip3 install --no-cache-dir --verbose -r requirements.txt && \
    python3 setup.py bdist_wheel

RUN cp faster-whisper/dist/faster_whisper*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/faster_whisper*.whl

RUN pip3 show faster_whisper && python3 -c 'import faster_whisper'