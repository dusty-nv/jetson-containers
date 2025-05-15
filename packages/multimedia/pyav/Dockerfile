#---
# name: pyav
# group: multimedia
# depends: [python, ffmpeg:git]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install PyAV (Python bindings for FFmpeg)
RUN pip3 install av
