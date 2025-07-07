#---
# name: wyoming-assist-microphone
# group: wyoming
# requires: '>=34.1.0'
# config: config.py
# docs: docs.md
# depends: [python, homeassistant-base]
# test: list_audio_devices.sh
# notes: The `wyoming-assist-microphone` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/tree/master/assist_microphone`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FORCE_BUILD=off \
    SATELLITE_VERSION \
    SATELLITE_BRANCH

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    SATELLITE_VERSION="$SATELLITE_VERSION" \
    SATELLITE_BRANCH="$SATELLITE_BRANCH" \
    SATELLITE_NAME="assist microphone" \
    SATELLITE_PORT="10700" \
    SATELLITE_SOUND_ENABLED="true" \
    SATELLITE_AWAKE_WAV="/usr/src/sounds/awake.wav" \
    SATELLITE_DONE_WAV="/usr/src/sounds/done.wav" \
    SATELLITE_DEBUG="true" \
    \
    ASSIST_PIPELINE_NAME="Home Assistant" \
    WAKEWORD_SERVICE_URI="tcp://127.0.0.1:10400" \
    WAKEWORD_NAME="ok_nabu" \
    \
    SATELLITE_SND_VOLUME_MULTIPLIER="1.0" \
    SATELLITE_MIC_VOLUME_MULTIPLIER="1.0" \
    SATELLITE_MIC_AUTO_GAIN="0" \
    SATELLITE_MIC_NOISE_SUPPRESSION="0" \
    \
    SATELLITE_AUDIO_DEVICE="plughw:CARD=S330,DEV=0"

COPY rootfs/ /
COPY *.sh /tmp/wyoming/assist-microphone/

WORKDIR /usr/src

RUN /tmp/wyoming/assist-microphone/install.sh || /tmp/wyoming/assist-microphone/build.sh

EXPOSE ${SATELLITE_PORT}/tcp

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost ${SATELLITE_PORT} \
        | grep -q "assist" \
        || exit 1
