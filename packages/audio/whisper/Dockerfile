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

RUN pip3 install pandas scipy jiwer ipywebrtc

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(f"torch.distributed:   {torch.distributed.is_available()}"); print(torch.__config__.show());'

# Clone the repository:
RUN git clone https://github.com/openai/whisper/ && \
    cd whisper && \
    sed 's|^numba.*||' -i requirements.txt && \
    sed 's|^numpy.*||' -i requirements.txt && \
    sed 's|^torch.*||' -i requirements.txt && \
    cat requirements.txt && \
    pip3 install .

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(f"torch.distributed:   {torch.distributed.is_available()}"); print(torch.__config__.show());'

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
