<p align="center"><img src="whisper.png" title="Wyoming whisper" alt="Wyoming whisper" style="width:100%;max-width:600px" /></p>

[`Home Assistant`](https://www.home-assistant.io/) add-on that uses [`wyoming-faster-whisper`](https://github.com/rhasspy/wyoming-faster-whisper) for speech-to-text system using the [`wyoming` protocol](https://www.home-assistant.io/integrations/wyoming/) on **NVIDIA Jetson** devices. Thank you to [**@ms1design**](https://github.com/ms1design) for contributing these Home Assistant & Wyoming containers!

## Features

- [x] Works well with [`home-assistant-core`](packages/smart-home/homeassistant-core) container on **Jetson devices** as well as Home Assistant hosted on different hosts
- [x] `GPU` accelerated on **Jetson Devices** thanks to [`faster-whisper` container](packages/audio/faster-whisper)

> Requires **Home Assistant** `2023.9` or later.

## `docker-compose` example

If you want to use `docker compose` to run [Home Assistant Core](/packages/smart-home/homeassistant-core/) [Voice Assistant Pipeline](https://www.home-assistant.io/voice_control/) on a **Jetson** device with `cuda` enabled, you can find a full example [`docker-compose.yaml` here](/packages/smart-home/wyoming/docker-compose.yaml).

```yaml
name: home-assistant-jetson
version: "3.9"
services:
  homeassistant:
    image: dustynv/homeassistant-core:latest-r36.2.0
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

  whisper:
    image: dustynv/wyoming-whisper:latest-r36.2.0
    restart: unless-stopped
    runtime: nvidia
    network_mode: host
    container_name: faster-whisper
    hostname: faster-whisper
    init: false
    ports:
      - "10300:10300/tcp"
    volumes:
      - hf-cache:/data/models/huggingface
      - ha-whisper-models:/data/models/faster-whisper
      - /etc/localtime:/etc/localtime:ro

volumes:
  ha-config:
  ha-whisper-models:
  hf-cache:
```

## Environment variables

| Variable | Type | Default | Description
| - | - | - | - |
| `WHISPER_PORT` | `str` | `10300` | Port number to use on `host` |
| `WHISPER_MODEL` | `str` | `tiny-int8` | Name of `faster-whisper` model to use (or `auto`) from [supported models list](https://github.com/home-assistant/addons/blob/master/whisper/config.yaml#L22) |
| `WHISPER_BEAM_SIZE` | `int` | `1` | Size of beam during decoding (0 for auto) |
| `WHISPER_LANGUAGE` | `str` | `en` | Default language to set for transcription from [supported languages list](https://github.com/home-assistant/addons/blob/master/whisper/config.yaml#L25) |
| `WHISPER_DEBUG` | `bool` | `true` | Log `DEBUG` messages |
| `WHISPER_COMPUTE_TYPE` | `str` | `default` | Compute type (`float16`, `int8`, etc.) |
| `WHISPER_INITIAL_PROMPT` | `str` | - | Optional text to provide as a prompt for the first window |
| `WHISPER_OFFLINE` | `bool` | `false` | Don't check HuggingFace hub for updates every time if set to `true` |

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

> [!NOTE]
> This project was created by [Jetson AI Lab Research Group](https://www.jetson-ai-lab.com/research.html).
