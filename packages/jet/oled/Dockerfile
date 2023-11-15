#---
# name: oled
# group: jet
# depends: [python3]
# notes: https://github.com/NVIDIA-AI-IOT/jetcard/
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


# ============================
# Enable OLED display
# ============================
RUN apt-get update && apt-get install -y unzip
RUN apt install -y python3-smbus python3-pil

RUN pip3 install pillow flask Adafruit_SSD1306

# oled_display_server.py
# utils.py

WORKDIR /opt

COPY oled_display_server.py /opt/oled/oled_display_server.py
COPY utils.py /opt/oled/utils.py
RUN cd oled  && \
    touch __init__.py 

CMD python3 /opt/oled/oled_display_server.py