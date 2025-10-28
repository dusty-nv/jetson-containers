#---
# name: gstreamer
# group: multimedia
# depends: opencv
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


# gstreamer1.0-plugins-bad depend on the ubuntu version of opencv, and will fail to install
# if opencv_cuda is already installed (so remove it and re-install it again after)
RUN apt-get update && \
    apt-get purge -y '*opencv*' || echo "existing OpenCV installation not found" && \
    apt-get install -y --no-install-recommends \
		  lsb-release \
      libunwind-dev \
		  libgstreamer1.0-dev \
		  gstreamer1.0-tools \
		  gstreamer1.0-libav \
		  gstreamer1.0-rtsp \
		  gstreamer1.0-plugins-good \
		  gstreamer1.0-plugins-bad \
		  gstreamer1.0-plugins-ugly \
		  libgstreamer-plugins-base1.0-dev \
		  libgstreamer-plugins-good1.0-dev \
		  libgstreamer-plugins-bad1.0-dev \
		  libgstrtspserver-1.0-0 \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-nice \
      libnss-mdns \
      avahi-utils \
		  python3-gi \
		  python3-gst-1.0 && \
    if [ `lsb_release --codename --short` != 'bionic' ]; then \
    apt-get install -y --no-install-recommends \
		  gstreamer1.0-plugins-rtp; \
    else echo "skipping packages unavailable for Ubuntu 18.04"; fi \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -s /usr/lib/python3/dist-packages/gi /opt/venv/lib/python3.*/site-packages/ # make gi available in the venv


# re-install opencv and make a copy of these gstreamer headers (some of them get purged)
RUN cp -r /usr/include/gstreamer-1.0 /tmp && \
    /tmp/opencv/install.sh && \
    cp -r /tmp/gstreamer-1.0 /usr/include && \
    rm -rf /tmp/gstreamer-1.0 && \
    ln -s /usr/lib/$(uname -m)-linux-gnu/libgstwebrtc-1.0.so.0 /usr/lib/$(uname -m)-linux-gnu/libgstwebrtc-1.0.so || \
    echo "libgstwebrtc already setup"
