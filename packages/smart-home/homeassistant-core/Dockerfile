#---
# name: homeassistant-core
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# docs: docs.md
# depends: [build-essential, ffmpeg]
# test: [test_ciso8601.py, test_psutil_home_assistant.py, test_homeassistant.py]
# notes: The `homeassistant-core` wheel that's build is saved in `/opt/`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HA_BRANCH

# Install the dependencies
RUN set -ex \
	&& apt-get update \
    && apt-get install -y --no-install-recommends \
        bluez \
        libpcap-dev \
        libffi-dev \
        libssl-dev \
        libjpeg-dev \
        zlib1g-dev \
        autoconf \
        libopenjp2-7 \
        libtiff5 \
        libturbojpeg0-dev \
        tzdata \
        liblapack3 \
        liblapack-dev \
        libatlas-base-dev \
	&& apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY *.diff /tmp/

# Install ciso8601
RUN set -ex \
    && git clone https://github.com/closeio/ciso8601 /tmp/ciso8601 \
    && git -C /tmp/ciso8601 apply /tmp/ciso8601-patch.diff \
    && git -C /tmp/ciso8601 diff \
    && pip3 wheel --wheel-dir=/opt/ --no-deps --verbose /tmp/ciso8601 \
    && pip3 install --no-cache-dir --verbose /opt/ciso8601*.whl \
    && pip3 show ciso8601 \
    && python3 -c 'import ciso8601; print(ciso8601.__version__);' \
    && rm -rf /tmp/ciso8601

# Install psutil-home-assistant
RUN set -ex \
    && git clone https://github.com/home-assistant-libs/psutil-home-assistant /tmp/psutil-home-assistant \
    && pip3 wheel --wheel-dir=/opt/ --no-deps --verbose /tmp/psutil-home-assistant \
    && pip3 install --no-cache-dir --verbose /opt/psutil_home_assistant*.whl \
    && pip3 show psutil-home-assistant \
    && python3 -c 'import psutil_home_assistant;' \
    && rm -rf /tmp/psutil-home-assistant

# Add an account for Home Assistant Core called `homeassistant`
RUN set -ex \
    && useradd -rm homeassistant \
    && mkdir /srv/homeassistant \
    && chown homeassistant:homeassistant /srv/homeassistant

# Compile & intall Home Assistant Core from source (`pypi` installation fails)
RUN set -ex \
    && echo "HA_BRANCH: ${HA_BRANCH}" \
    && python3 --version \
    && git clone --branch=${HA_BRANCH} https://github.com/home-assistant/core /opt/home-assistant \
    && pip3 install --no-cache-dir --ignore-installed blinker \
    && pip3 install --no-cache-dir --verbose -r /opt/home-assistant/requirements_all.txt \
    && pip3 wheel --wheel-dir=/opt --verbose /opt/home-assistant \
    && pip3 install --no-cache-dir --verbose /opt/homeassistant*.whl \
    && pip3 show homeassistant \
    && python3 -c 'import homeassistant;' \
    && rm -rf /opt/home-assistant

USER homeassistant

CMD ["hass"]
