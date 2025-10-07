#---
# name: voice-pro
# group: audio
# docs: docs.md
# depends: [cuda, pytorch, torchvision, torchaudio, torchao, torch-memory-saver, audiocraft, whisper, whisperx]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


WORKDIR /opt

# Install dependencies"
RUN apt-get update && \
    apt-get install -y --no-install-recommends espeak-ng && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get install -y espeak espeak-data libespeak1 libespeak-dev && \
    apt-get install -y festival* && \
    apt-get install -y build-essential && \
    apt-get install -y flac libasound2-dev libsndfile1-dev vorbis-tools && \
    apt-get install -y libxml2-dev libxslt-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*  && \
    apt-get clean

RUN uv pip install phonemizer  && \
    uv pip install torchmetrics

# Clone the repository:
RUN git clone --recursive depth=1 https://github.com/abus-aikorea/voice-pro.git /opt/voice-pro && \
    cd /opt/voice-pro && \
    sed -i '/torchvision==0.20.1+cu124/d' requirements-voice-gpu.txtrequirements.txt && \
    sed -i '/torchaudio==2.5.1+cu124/d' requirements-voice-gpu.txt && \
    sed -i '/--extra-index-url https:\/\/download.pytorch.org\/whl\/cu124/d' requirements-voice-gpu.txt && \
    sed -i 's/==/>=/g' requirements-voice-gpu.txt && \
    sed -i 's/~=/>=/g' requirements-voice-gpu.txt && \
    uv pip install -U -requirements-voice-gpu.txt

WORKDIR /opt/voice-pro/

CMD [ "python3", "start-abus.py" , "voice"]
