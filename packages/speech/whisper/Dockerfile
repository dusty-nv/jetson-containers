#---
# name: whisper
# group: audio
# depends: [numba, numpy, pytorch, torchaudio, jupyterlab]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN uv pip install pandas scipy jiwer ipywebrtc

ADD https://api.github.com/repos/openai/whisper/git/refs/heads/main /tmp/whisper_version.json

# Clone the repository:
RUN git clone https://github.com/openai/whisper/ && \
    cd whisper && \
    sed -i 's/==/>=/g' requirements.txt && \
    sed -i 's/~=/>=/g' requirements.txt && \
    cat requirements.txt && \
    uv build --wheel --out-dir dist --no-deps --verbose .

RUN cp whisper/dist/openai_whisper*.whl /opt && \
    uv pip install /opt/openai_whisper*.whl

WORKDIR /opt/whisper/

COPY record-and-transcribe.ipynb /opt/whisper/notebooks

RUN openssl req \
        -new \
        -newkey rsa:4096 \
        -days 365 \
        -nodes \
        -x509 \
        -keyout mykey.key \
        -out mycert.pem \
        -subj '/CN=localhost'

CMD /bin/bash -c "jupyter lab --ip 0.0.0.0 --port 8888  --certfile=mycert.pem --keyfile mykey.key --allow-root &> /var/log/jupyter.log" & \
	echo "allow 10 sec for JupyterLab to start @ https://$(hostname -I | cut -d' ' -f1):8888 (password nvidia)" && \
	echo "JupterLab logging location:  /var/log/jupyter.log  (inside the container)" && \
	/bin/bash
