#---
# name: oled
# group: hardware
# depends: []
# notes: https://github.com/NVIDIA-AI-IOT/jetcard/
#---
ARG BASE_IMAGE
FROM ubuntu:24.04

# Avoid warnings by switching to noninteractive
ARG DEBIAN_FRONTEND=noninteractive

# ============================
# Enable OLED display
# ============================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  build-essential \
		  python3-dev \
          python3-pip \
          python3-pil \
		  python3-smbus \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN uv pip install pillow flask Adafruit_SSD1306 smbus2

### TODO: 232 MB of disk space should be freed, but not on the resulting container image
RUN apt-get purge -y \
		  build-essential \
		  python3-dev && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

WORKDIR /opt

COPY oled_server.py /opt/oled/oled_server.py
COPY utils.py /opt/oled/utils.py
RUN cd oled  && \
    touch __init__.py

COPY start_oled_server /
CMD /start_oled_server && /bin/bash
#CMD /bin/bash
