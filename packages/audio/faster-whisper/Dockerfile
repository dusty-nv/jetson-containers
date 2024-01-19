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

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
# 		  ffmpeg \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# RUN pip3 install scipy jiwer ipywebrtc

# Clone the repository:
RUN git clone https://github.com/guillaumekln/faster-whisper && \
    cd faster-whisper && \
    # sed 's|^onnxruntime.*||' -i requirements.txt && \
    sed 's|^ctranslate2.*||' -i requirements.txt && \
    sed 's|^huggingface_hub.*||' -i requirements.txt && \
    cat requirements.txt && \
    pip3 install .

WORKDIR /opt/faster-whisper/