#---
# name: faster-whisper
# group: audio
# depends: [ctranslate2, huggingface_hub, numpy]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN git clone https://github.com/guillaumekln/faster-whisper && \
    cd faster-whisper && \
    # sed 's|^onnxruntime.*||' -i requirements.txt && \
    sed 's|^ctranslate2.*||' -i requirements.txt && \
    sed 's|^huggingface_hub.*||' -i requirements.txt && \
    cat requirements.txt && \
    pip3 install --no-cache-dir --verbose -r requirements.txt && \
    python3 setup.py bdist_wheel

RUN cp faster-whisper/dist/faster_whisper*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/faster_whisper*.whl

RUN pip3 show faster_whisper && python3 -c 'import faster_whisper'