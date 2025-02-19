#---
# name: kokoro-tts:hf
# group: audio
# depends: [python, cuda, pytorch, transformers, torchaudio]
# test: [test.py]
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install required dependencies
RUN apt-get update && \
    apt-get install -y libespeak-ng-dev portaudio19-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install specific versions of Python packages
RUN pip3 install \
    scipy==1.15.1 \
    loguru==0.7.3 \
    num2words==0.5.14 \
    spacy==3.5.0 \
    phonemizer==3.3.0 \
    rfc3986==1.5.0 \
    IPython==8.32.0 \ 
    pydantic==1.10.21 \
    espeakng-loader


RUN pip3 install misaki --no-deps
RUN pip3 install kokoro --no-deps
    
# Install additional required packages
#RUN pip3 install gradio soundfile sounddevice

# Install specific versions of Python packages
#RUN pip3 install \
#    pydantic==1.10.21



# Reinstall correct numpy
#RUN bash /tmp/numpy/install.sh

# Restore working directory
WORKDIR /

# Default command
CMD ["bash"]
