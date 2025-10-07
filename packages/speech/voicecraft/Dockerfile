#---
# name: voicecraft
# group: audio
# docs: docs.md
# depends: [audiocraft, whisperx]
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
RUN git clone https://github.com/jasonppy/VoiceCraft.git && \
    cd VoiceCraft && \
    sed 's|^whisperx.*||' -i gradio_requirements.txt && \
    sed 's|^huggingface_hub.*||' -i gradio_requirements.txt && \
    cat gradio_requirements.txt && \
    pip install -r gradio_requirements.txt

WORKDIR /opt/VoiceCraft/

CMD [ "python3", "./gradio_app.py" , "--server_name" , "0.0.0.0" ]
