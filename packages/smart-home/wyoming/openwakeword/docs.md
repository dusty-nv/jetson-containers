## Wyoming `openWakeWord`

<p align="center"><img src="wyoming-openwakeword.png" title="Wyoming openWakeWord" alt="Wyoming openWakeWord" /></p>

[`Home Assistant`](https://www.home-assistant.io/) add-on that uses [`openWakeWord`](https://github.com/dscripka/openwakeword) for wake word detection over the [Wyoming protocol](https://www.home-assistant.io/integrations/wyoming/).

Requires **Home Assistant** `2023.9` or later.

### `docker-compose` example

```yaml
name: home-assistant
version: "3.9"
services:
  home-assistant:
    image: dustynv/homeassistant-core:r36.2.0
    restart: unless-stopped
    runtime: nvidia
    privileged: true
    network_mode: host
    container_name: home-assistant
    hostname: home-assistant
    ports:
      - "8123:8123"
    devices:
      - /dev/snd:/dev/snd
      - /dev/bus/usb
    volumes:
      - ha-config:/config
      - /etc/localtime:/etc/localtime:ro
      - /etc/timezone:/etc/timezone:ro
    environment:
      TZ: Europe/Amsterdam
    stdin_open: true
    tty: true
    healthcheck:
      test: curl -s -o /dev/null -w "%{http_code}" http://localhost:8123 || exit 1
      interval: 1m
      timeout: 30s
      retries: 3

  openwakeword:
    image: dustynv/wyoming-openwakeword:r36.2.0
    restart: unless-stopped
    runtime: nvidia
    network_mode: host
    container_name: openwakeword
    hostname: openwakeword
    depends_on:
      home-assistant:
        condition: service_healthy
    ports:
      - "10400:10400/tcp"
    devices:
      - /dev/snd:/dev/snd
      - /dev/bus/usb
    volumes:
      - ha-openwakeword-models:/share/openwakeword
      - /etc/localtime:/etc/localtime:ro
      - /etc/timezone:/etc/timezone:ro
    environment:
      TZ: Europe/Amsterdam
    stdin_open: true
    tty: true

volumes:
  ha-config:
  ha-openwakeword-models:
```

### Native auto-discovery with Home Assistant

> **TLDR;** *It's disabled, go with manual way...*

The native auto-discovery of add-ons running on the same host/network is disabled due to the requirement of running [`Home Assistant Supervisor`](https://www.home-assistant.io/integrations/hassio/). This has some deep debian system dependencies which are too tidious to port.

Most Home Assistant add-on's are using [`bashio`](https://github.com/hassio-addons/bashio) under the hood so some of the system overlays commands ware adjusted to make it work without `Supervisor`.

### Manual discovery

To add the `wyoming-openwakeword` add-on to the running Home Assistant instance manually, just follow the below steps:

1. Browse to your Home Assistant instance (`homeassistant.local:8123`).
2. Go to `Settings > Devices & Services`.
3. In the bottom right corner, select the `Add Integration` button.
4. From the list, search & select `Wyoming Protocol`.
5. Enter the `wyoming-openwakeport` Host IP address (use `localhost` if running of the same host as Home Assistant).
6. Enter the `wyoming-openwakeport` port (default is `10400`).

### Configure the Home Assistant Assist

<p align="center"><img src="openwakeword-assist-config.png" title="Wyoming openWakeWord" alt="Wyoming openWakeWord" /></p>

Read more how to configure `wyoming-openwakeword` in the [official documentation](https://www.home-assistant.io/voice_control/install_wake_word_add_on#enabling-wake-word-for-your-voice-assistant).