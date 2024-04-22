#---
# name: wyoming-assist-microphone
# group: wyoming
# requires: '>=34.1.0'
# config: config.py
# depends: [homeassistant-base, python:3.11]
# test: [list_audio_devices.sh]
# notes: The `assist-microphone` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/tree/master/assist_microphone`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG WYOMING_SATELLITE_VERSION

# Specify the shell with desired options
SHELL ["/bin/bash", "-euxo", "pipefail"]

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    WYOMING_SATELLITE_VERSION="$WYOMING_SATELLITE_VERSION" \
    ASSIST_MICROPHONE_PORT="10700" \
    ASSIST_MICROPHONE_SOUND_ENABLED="true" \
    ASSIST_MICROPHONE_AWAKE_WAV="/usr/src/sounds/awake.wav" \
    ASSIST_MICROPHONE_DONE_WAV="/usr/src/sounds/done.wav" \
    ASSIST_MICROPHONE_DEBUG="true" \
    WAKEWORD_SERVICE_URI="tcp://127.0.0.1:10400" \
    WAKEWORD_NAME="ok_nabu" \
    \
    ASSIST_MICROPHONE_SND_COMMAND="aplay -r 16000 -c 1 -f S16_LE -t raw" \
    ASSIST_MICROPHONE_SND_COMMAND_RATE="16000" \
    ASSIST_MICROPHONE_SND_VOLUME_MULTIPLIER="1.0" \
    \
    ASSIST_MICROPHONE_MIC_COMMAND="arecord -r 16000 -c 1 -f S16_LE -t raw" \
    ASSIST_MICROPHONE_MIC_COMMAND_RATE="16000" \
    ASSIST_MICROPHONE_MIC_VOLUME_MULTIPLIER="1.0" \
    ASSIST_MICROPHONE_MIC_AUTO_GAIN="0" \
    ASSIST_MICROPHONE_MIC_NOISE_SUPPRESSION="0"

COPY rootfs/ /
COPY build.sh /tmp/wyoming/build_wyoming-assist-microphone.sh
COPY list_audio_devices.py test_recording_and_playing.sh test_speaker.py /tmp/

WORKDIR /usr/src

RUN /tmp/wyoming/build_wyoming-assist-microphone.sh

EXPOSE ${ASSIST_MICROPHONE_PORT}/tcp

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost ${ASSIST_MICROPHONE_PORT} \
        | grep -q "assist" \
        || exit 1

# Disabled due to issues when running `jetson-containers` tests. It's more recomended
# to use `command: "/init"` when `docker-compose` is used or just `--entrypoint "/init"`
# when using `docker run`.
# ENTRYPOINT ["/init"]
