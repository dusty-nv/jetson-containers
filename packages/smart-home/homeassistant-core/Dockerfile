#---
# name: homeassistant-core
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# docs: docs.md
# depends: [build-essential, homeassistant-base-image, ffmpeg]
# test: [test_ciso8601.py, test_psutil_home_assistant.py, test_homeassistant.py]
# notes: The `homeassistant-core` wheel that's build is saved in `/usr/src/homeassistant`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HA_BRANCH

ENV S6_SERVICES_READYTIME=50 \
    S6_SERVICES_GRACETIME=240000 \
    UV_SYSTEM_PYTHON=true

# Base system
WORKDIR /usr/src

COPY *.diff /tmp/

# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf \
        libpcap0.8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip3 install --no-cache-dir --verbose uv==0.1.27

# Install ciso8601
RUN git clone https://github.com/closeio/ciso8601 /tmp/ciso8601 \
    && git -C /tmp/ciso8601 apply /tmp/ciso8601-patch.diff \
    && git -C /tmp/ciso8601 diff \
    && pip3 wheel --wheel-dir=/opt/ --no-deps --verbose /tmp/ciso8601 \
    && pip3 install --no-cache-dir --verbose /opt/ciso8601*.whl \
    && pip3 show ciso8601 \
    && python3 -c 'import ciso8601; print(ciso8601.__version__);' \
    && rm -rf /tmp/ciso8601

# Install psutil-home-assistant
RUN git clone https://github.com/home-assistant-libs/psutil-home-assistant /tmp/psutil-home-assistant \
    && pip3 wheel --wheel-dir=/opt/ --no-deps --verbose /tmp/psutil-home-assistant \
    && pip3 install --no-cache-dir --verbose /opt/psutil_home_assistant*.whl \
    && pip3 show psutil-home-assistant \
    && python3 -c 'import psutil_home_assistant;' \
    && rm -rf /tmp/psutil-home-assistant

RUN echo "HA_BRANCH: ${HA_BRANCH}" \
    && python3 --version \
    && gcc --version \
    && g++ --version \
    && pip3 install --no-cache-dir --ignore-installed blinker \
    && git clone --branch=${HA_BRANCH} https://github.com/home-assistant/core homeassistant \
    \
    # Fix: https://community.home-assistant.io/t/new-install-onboarding-failedkeyerror-component-onboarding-area-living-room/689712 \
    # && git -C homeassistant apply /tmp/onboarding-translations-patch.diff \
    # && git -C homeassistant diff \
    \
    && uv pip --version \
    && uv pip install --no-cache-dir --verbose -r homeassistant/requirements.txt

RUN if ls homeassistant/home_assistant_*.whl 1> /dev/null 2>&1; then \
        uv pip install --no-cache-dir --verbose homeassistant/home_assistant_*.whl; \
    fi \
    && ln -s /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 /usr/local/lib/libjemalloc.so.2 \
    && if [ "${BUILD_ARCH}" = "i386" ]; then \
        LD_PRELOAD="${LD_PRELOAD}:/usr/local/lib/libjemalloc.so.2" \
        MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000" \
        linux32 uv pip install \
            --no-cache-dir \
            --verbose \
            -r homeassistant/requirements_all.txt; \
    else \
        LD_PRELOAD="${LD_PRELOAD}:/usr/local/lib/libjemalloc.so.2" \
        MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000" \
        uv pip install \
            --no-cache-dir \
            --verbose \
            -r homeassistant/requirements_all.txt; \
    fi \
    && uv pip install -e homeassistant \
    && python3 -m compileall homeassistant \
    && python3 -m script.hassfest \
    && python3 -m script.translations develop --all

# Home Assistant S6-Overlay
RUN cp -r homeassistant/rootfs/* /

WORKDIR /config

ENTRYPOINT ["/init"]
