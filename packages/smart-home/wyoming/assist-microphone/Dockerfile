#---
# name: wyoming-assist-microphone
# group: wyoming
# requires: '>=34.1.0'
# depends: [homeassistant-base, python:3.11]
# notes: The `assist-microphone` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/tree/master/assist_microphone`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Specify the shell with desired options
SHELL ["/bin/bash", "-euxo", "pipefail"]

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    WYOMING_SATELLITE_VERSION="1.2.0" \
    ASSIST_MICROPHONE_PORT="10700" \
    ASSIST_MICROPHONE_SOUND_ENABLED="true" \
    ASSIST_MICROPHONE_AWAKE_WAV="/usr/src/sounds/awake.wav" \
    ASSIST_MICROPHONE_DONE_WAV="/usr/src/sounds/done.wav" \
    ASSIST_MICROPHONE_MIC_VOLUME_MULTIPLIER="1.0" \
    ASSIST_MICROPHONE_SND_VOLUME_MULTIPLIER="1.0" \
    ASSIST_MICROPHONE_MIC_AUTO_GAIN="0" \
    ASSIST_MICROPHONE_MIC_NOISE_SUPPRESSION="0" \
    ASSIST_MICROPHONE_DEBUG="true"

COPY build.sh /tmp/wyoming/build_wyoming-assist-microphone.sh

WORKDIR /usr/src

RUN /tmp/wyoming/build_wyoming-assist-microphone.sh

EXPOSE ${ASSIST_MICROPHONE_PORT}/tcp

ENTRYPOINT ["/init"]
