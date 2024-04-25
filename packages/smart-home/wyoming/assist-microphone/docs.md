## Wyoming `assist-microphone`

<p align="center"><img src="images/wyoming-voice-assistant.png" title="Wyoming assist-microphone" alt="Wyoming assist-microphone" /></p>

[`Home Assistant`](https://www.home-assistant.io/) add-on that uses [`wyoming-satellite`](https://github.com/rhasspy/wyoming-satellite) for remote voice satellite using the [`wyoming` protocol](https://www.home-assistant.io/integrations/wyoming/) on **NVIDIA Jetson** devices.

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

  assist-microphone:
    image: ms1design/wyoming-assist-microphone:latest-r36.2.0-cu122-cp311
    restart: unless-stopped
    network_mode: host
    container_name: assist-microphone
    hostname: assist-microphone
    runtime: nvidia
    init: false
    ports:
      - "10700:10700/tcp"
    devices:
      - /dev/snd:/dev/snd
      - /dev/bus/usb
    volumes:
      - ha-assist-microphone:/share
      - /etc/localtime:/etc/localtime:ro
      - /etc/timezone:/etc/timezone:ro
    environment:
      AUDIO_DEVICE: "plughw:CARD=S330,DEV=0"
      ASSIST_MICROPHONE_SND_VOLUME_MULTIPLIER: 0.3
      WAKEWORD_NAME: "ok_nabu"
      ASSIST_PIPELINE_NAME: "Home Assistant"
    stdin_open: true
    tty: true

volumes:
  ha-config:
  ha-assist-microphone:
```
</details>

### Environment variables

| Variable | Type | Default | Description
| - | - | - | - |
| `OPENWAKEWORD_PORT` | `str` | `10400` | Port number to use on `host` |
| `OPENWAKEWORD_THRESHOLD` | `float` | `0.5` | Wake word model threshold (`0.0`-`1.0`), where higher means fewer activations. |
| `OPENWAKEWORD_TRIGGER_LEVEL` | `int` | `1` | Number of activations before a detection is registered. A higher trigger level means fewer detections. |
| `OPENWAKEWORD_PRELOAD_MODEL`| `str` | `ok_nabu` | Name or path of wake word model(s) to pre-load |
| `OPENWAKEWORD_CUSTOM_MODEL_DIR` | `str` | `/share/openwakeword` | Path to directory with custom wake word models |
| `OPENWAKEWORD_DEBUG` | `bool` | `true` | Log `DEBUG` messages |

## Configuration

Read more how to configure `wyoming-openwakeword` in the [official documentation](https://www.home-assistant.io/voice_control/voice_remote_local_assistant):

<p align="center"><img src="images/configuration.png" title="Wyoming assist-microphone" alt="Wyoming assist-microphone" /></p>

## TODO's

- [ ] Build `openWakeWord` from source based on `onnxruntime` `gpu` enabled container (currently `openWakeWord` is still using `tflite` models instead `onnx`)
- [ ] Custom Wake Word Models training container using automatic synthetic data creation

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
