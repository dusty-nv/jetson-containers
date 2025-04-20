#---
# name: pyav
# group: multimedia
# depends: [python, ffmpeg:git]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# most of this dockerfile was to build ffmpeg
# with AV1 enabled, and that was moved to ffmpeg
RUN pip3 install pyav
