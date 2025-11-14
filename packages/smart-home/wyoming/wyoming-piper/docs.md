<p align="center"><img src="images/piper.png" title="Wyoming piper" alt="Wyoming piper" style="width:100%;max-width:600px" /></p>

[`Home Assistant`](https://www.home-assistant.io/) add-on that uses [`wyoming-piper`](https://github.com/rhasspy/wyoming-piper) for text-to-speech system using the [`wyoming` protocol](https://www.home-assistant.io/integrations/wyoming/) on **NVIDIA Jetson** devices. Thank you to [**@ms1design**](https://github.com/ms1design) for contributing these Home Assistant & Wyoming containers!

## Features

- [x] Works well with [`home-assistant-core`](packages/smart-home/homeassistant-core) container on **Jetson** devices as well as Home Assistant hosted on different hosts
- [x] `GPU` accelerated on **Jetson** devices thanks to [`piper-tts` container](packages/audio/piper-tts)

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

  piper-tts:
    image: dustynv/wyoming-piper:master-r36.2.0
    restart: unless-stopped
    network_mode: host
    container_name: piper-tts
    hostname: piper-tts
    runtime: nvidia
    init: false
    ports:
      - "10200:10200/tcp"
    devices:
      - /dev/snd:/dev/snd
      - /dev/bus/usb
    volumes:
      - ha-piper-tts-models:/data/models/piper
      - /etc/localtime:/etc/localtime:ro
      - /etc/timezone:/etc/timezone:ro

volumes:
  ha-config:
  ha-piper-tts-models:
```

### Environment variables

| Variable | Type | Default | Description
| - | - | - | - |
| `PIPER_PORT` | `str` | `10200` | Port number to use on `host` |
| `PIPER_CACHE` | `str` | `/data/models/piper` | Name of a default mounted location available for downloading the models |
| `PIPER_UPDATE_VOICES` | `bool` | `true` | Download latest `voices.json` during startup |
| `PIPER_LENGTH_SCALE` | `float` | `1.0` | Phoneme length |
| `PIPER_NOISE_SCALE` | `float` | `0.667` | Generator noise |
| `PIPER_NOISE_W` | `float` | `0.333` | Phoneme width noise |
| `PIPER_SPEAKER` | `[str, int]` | `0` | Name or id of speaker for default voice |
| `PIPER_VOICE` | `str` | `en_US-lessac-high` | Default Piper voice to use (e.g., `en_US-lessac-medium`) from [list of available voices](https://github.com/rhasspy/piper?tab=readme-ov-file#voices)
| `PIPER_MAX_PROC` | `int` | `1` | Maximum number of piper process to run simultaneously |
| `PIPER_DEBUG` | `bool` | `true` | Log `DEBUG` messages |
| `PIPER_USE_CUDA` | `bool` | `true` | Enable CUDA |

## Configuration

Read more how to configure `wyoming-piper` in the [official documentation](https://www.home-assistant.io/voice_control/voice_remote_local_assistant#installing-a-local-assist-pipeline):

<p align="center"><img src="images/piper-playground.png" title="Wyoming assist-microphone" alt="Wyoming assist-microphone" style="width:100%;max-width:600px" /></p>

### Available Voices

List of available voices to download is [available here](https://github.com/rhasspy/piper?tab=readme-ov-file#voices).

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