# openwakeword

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

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

> **TLDR;** *It's didabled, go with manual way...*

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
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`wyoming-openwakeword:latest`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `wyoming-openwakeword` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`homeassistant-base`](/packages/smart-home/homeassistant-base) [`python:3.11`](/packages/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The `openWakeWord` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/blob/master/openwakeword/Dockerfile` |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use the [`run.sh`](/docs/run.md)/[`autotag`](/docs/run.md#autotag) helpers or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag openwakeword)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host openwakeword:36.2.0

```
> <sup>[`run.sh`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag openwakeword)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
./run.sh $(./autotag openwakeword) my_app --abc xyz
```
You can pass any options to [`run.sh`](/docs/run.md) that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
./build.sh openwakeword
```
The dependencies from above will be built into the container, and it'll be tested during.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
