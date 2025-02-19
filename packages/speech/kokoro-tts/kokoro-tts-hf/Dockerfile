#---
# name: kokoro-tts:hf
# group: audio
# depends: [transformers, torchaudio]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y libespeak-ng-dev portaudio19-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# pin these versions because of an infinite build loop with blis
# https://github.com/explosion/cython-blis/issues/117#issuecomment-2660947800
RUN BLIS_ARCH="generic" \
    pip3 wheel --wheel-dir=/opt/wheels --verbose \
        scipy==1.15.1 \
        loguru==0.7.3 \
        num2words==0.5.14 \
        spacy==3.5.0 \
        phonemizer==3.3.0 \
        rfc3986==1.5.0 \
        IPython==8.32.0 \ 
        pydantic==1.10.21 \
        ${NUMPY_PACKAGE} \
        espeakng-loader

RUN pip3 install --no-cache-dir --verbose /opt/wheels/*.whl

RUN git clone https://github.com/hexgrad/misaki /opt/misaki && \
    cd /opt/misaki && \
    pip3 install --no-deps --verbose -e .

RUN git clone https://github.com/hexgrad/kokoro /opt/kokoro && \
    cd /opt/kokoro && \
    pip3 install --no-deps --verbose -e .
