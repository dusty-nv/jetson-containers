<p align="center"><img src="whisper.png" title="Wyoming whisper" alt="Wyoming whisper" /></p>

[`Home Assistant`](https://www.home-assistant.io/) add-on that uses [`wyoming-faster-whisper`](https://github.com/rhasspy/wyoming-faster-whisper) for speech to text system using the [`wyoming` protocol](https://www.home-assistant.io/integrations/wyoming/) on **NVIDIA Jetson** devices.

### Features

- [x] Works well with [`home-assistant-core`](packages/smart-home/homeassistant-core) container on **Jetson devices** as well as Home Assistant hosted on different host's
- [x] `GPU` Accelerated on **Jetson Devices** thank's to [`faster-whisper` container](packages/audio/faster-whisper)

> Requires **Home Assistant** `2023.9` or later.

<details open>
<summary><h3 style="display:inline"><code>docker-compose</code> example</h3></summary>
<br>

```yaml
name: home-assistant-jetson
version: "3.9"
services:
  homeassistant:
    image: dusty-nv/homeassistant-core:latest-r36.2.0-cu122-cp310
    restart: unless-stopped
    init: false
    privileged: true
    network_mode: host
    container_name: homeassistant
    hostname: homeassistant
    ports:
      - "8123:8123"
    volumes:
      - ha-config:/config
      - /etc/localtime:/etc/localtime:ro
      - /etc/timezone:/etc/timezone:ro
    stdin_open: true
    tty: true

  whisper:
    image: dusty-nv/wyoming-whisper:latest-r36.2.0-cu122-cp311
    restart: unless-stopped
    runtime: nvidia
    network_mode: host
    container_name: faster-whisper
    hostname: faster-whisper
    init: false
    ports:
      - "10300:10300/tcp"
    volumes:
      - ha-whisper-models:/share/whisper
      - ha-whisper-data:/data
      - /etc/localtime:/etc/localtime:ro
      - /etc/timezone:/etc/timezone:ro
    environment:
      TZ: ${ENV_TZ}
    stdin_open: true
    tty: true

volumes:
  ha-config:
  ha-whisper-models:
  ha-whisper-data:
```
</details>

### Environment variables

| Variable | Type | Default | Description
| - | - | - | - |
| `WHISPER_PORT` | `str` | `10300` | Port number to use on `host` |
| `WHISPER_MODEL` | `str` | `tiny-int8` | Name of `faster-whisper` model to use from [supported models list](https://github.com/home-assistant/addons/blob/master/whisper/config.yaml#L22) |
| `WHISPER_BEAM_SIZE` | `int` | `1` | Beam size |
| `WHISPER_LANGUAGE` | `str` | `en` | Default language to set for transcription from [supported languages list](https://github.com/home-assistant/addons/blob/master/whisper/config.yaml#L25) |
| `WHISPER_DEBUG` | `bool` | `true` | Log `DEBUG` messages |

## Configuration

Read more how to configure `wyoming-whisper` in the [official documentation](https://www.home-assistant.io/voice_control/voice_remote_local_assistant#installing-a-local-assist-pipeline):

## TODO's

- [ ] Testing

## Support

Got questions? You have several options to get them answered:

#### For general **Home Assistant** Support:
- The [Home Assistant Discord Chat Server](https://discord.gg/c5DvZ4e).
- The Home Assistant [Community Forum](https://community.home-assistant.io/).
- Join the [Reddit subreddit](https://reddit.com/r/homeassistant) in [`/r/homeassistant`](https://reddit.com/r/homeassistant)
- In case you've found an bug in Home Assistant, please [open an issue on our GitHub](https://github.com/home-assistant/addons/issues).

#### For NVIDIA Jetson based Home Assistant Support:
- The NVIDIA Jetson AI Lab [tutorials section](https://www.jetson-ai-lab.com/tutorial-intro.html).
- The Jetson AI Lab - Home Assistant Integration [thread on NVIDIA's Developers Forum](https://forums.developer.nvidia.com/t/jetson-ai-lab-home-assistant-integration/288225).
- In case you've found an bug in `jetson-containers`, please [open an issue on our GitHub](https://github.com/dusty-nv/jetson-containers/issues).
