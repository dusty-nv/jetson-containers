#---
# name: sdnext
# group: diffusion
# depends: [pytorch, torchvision, torchaudio, opencv, bitsandbytes, transformers, xformers, torchao, flash-attention, sage-attention, sparge-attention, flexprefill, paraattention, diffusers, huggingface_hub]
# requires: '>=35.0.0'
# notes: https://github.com/vladmandic/sdnext
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/vladmandic/sdnext /opt/sdnext && \
    cd /opt/sdnext && \
    sed -i 's/==/>=/g' requirements.txt && \
    uv pip install -r requirements.txt

ENV SD_DATADIR="/mnt/data"
ENV SD_MODELSDIR="/mnt/models"

WORKDIR /opt/sdnext
# tcmalloc is not required but it is highly recommended
ENV LD_PRELOAD=libtcmalloc.so.4

RUN ["python3", "launch.py", "--debug", "--uv", "--use-cuda", "--log", "sdnext.log", "--test", "--optional"]
# preinstall additional packages to avoid installation during runtime

# expose port
EXPOSE 7860

# actually run sdnext
CMD ["python3", "launch.py", "--debug", "--skip-all", "--listen", "--quick", "--api-log", "--log", "sdnext.log"]

# healthcheck function
# HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 CMD curl --fail http://localhost:7860/sdapi/v1/status || exit 1


