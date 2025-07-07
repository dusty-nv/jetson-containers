# wyoming-assist-microphone

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<p align="center"><img src="images/wyoming-voice-assistant.png" style="width:100%;max-width:600px" title="Wyoming assist-microphone" alt="Wyoming assist-microphone" /></p>

[`Home Assistant`](https://www.home-assistant.io/) add-on that uses [`wyoming-satellite`](https://github.com/rhasspy/wyoming-satellite) for remote voice [satellite](https://www.home-assistant.io/integrations/wyoming#satellites) using the [`wyoming` protocol](https://www.home-assistant.io/integrations/wyoming/) on **NVIDIA Jetson** devices. Thank you to [**@ms1design**](https://github.com/ms1design) for contributing these Home Assistant & Wyoming containers!

## Features

- [x] Works well with [`home-assistant-core`](/packages/smart-home/homeassistant-core) container on **Jetson devices** as well as Home Assistant hosted on different hosts
- [x] Uses the [`wyoming-openwakeword`](/packages/smart-home/wyoming/openwakeword) container to detect wake words
- [x] Uses the [`wyoming-whisper`](/packages/smart-home/wyoming/wyoming-whisper) container to handle `STT`
- [x] Uses the [`wyoming-piper`](/packages/smart-home/wyoming/piper) container to handle `TTS`

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

  assist-microphone:
    image: dustynv/wyoming-assist-microphone:latest-r36.2.0
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
      SATELLITE_AUDIO_DEVICE: "plughw:CARD=S330,DEV=0"
      SATELLITE_SND_VOLUME_MULTIPLIER: 0.3
      WAKEWORD_NAME: "ok_nabu"
      ASSIST_PIPELINE_NAME: "Home Assistant"

volumes:
  ha-config:
  ha-assist-microphone:
```

## Environment variables

| Variable | Type | Default | Description
| - | - | - | - |
| `SATELLITE_NAME` | `str` | `assist microphone` | Name of the satellite |
| `SATELLITE_AUDIO_DEVICE` | `str` | `plughw:CARD=S330,DEV=0` | Selected Audio Device to use, [read more here](#determine-audio-devices) |
| `SATELLITE_PORT` | `str` | `10700` | Port of the satellite |
| `SATELLITE_SOUND_ENABLED` | `bool` | `true` | Enable or disable connected Speaker |
| `SATELLITE_AWAKE_WAV` | `str` | `/usr/src/sounds/awake.wav` | `WAV` file to play when wake word is detected |
| `SATELLITE_DONE_WAV` | `str` | `/usr/src/sounds/done.wav` | `WAV` file to play when voice command is done |
| `ASSIST_PIPELINE_NAME` | `str` | `Home Assistant` | Home Assistant Voice Assistant Pipeline name to run |
| `WAKEWORD_SERVICE_URI` | `str` | `tcp://127.0.0.1:10400` | `URI` of Wyoming wake word detection service |
| `WAKEWORD_NAME` | `str` | `ok_nabu` | Name of wake word to listen for |
| `SATELLITE_SND_VOLUME_MULTIPLIER` | `float` | `1.0` | Sound volume multiplier |
| `SATELLITE_MIC_VOLUME_MULTIPLIER` | `float` | `1.0` | Mic volume multiplier |
| `SATELLITE_MIC_AUTO_GAIN` | `int` | `0` | Mic auto gain |
| `SATELLITE_MIC_NOISE_SUPPRESSION` | `int` | `0` | Mic noise suppression (`0-4`) |
| `SATELLITE_DEBUG` | `bool` | `true` | Log `DEBUG` messages |

## Configuration

Read more how to configure `wyoming-assist-microphone` in the [official documentation](https://www.home-assistant.io/voice_control/voice_remote_local_assistant#installing-a-local-assist-pipeline).

<p align="center"><img src="images/configuration.png" title="Wyoming assist-microphone" alt="Wyoming assist-microphone" style="width:100%;max-width:600px" /></p>

### Determine Audio Devices

Picking the correct microphone/speaker devices is critical for the satellite to work.

List your available microphones with:
```bash
arecord -L
```

List your available speakers with:
```bash
aplay -L
```

You should see similar output to below for both commands:
```bash
plughw:CARD=seeed2micvoicec,DEV=0
    seeed-2mic-voicecard, bcm2835-i2s-wm8960-hifi wm8960-hifi-0
    Hardware device with all software conversions
```

Prefer ones that start with `plughw:` or just use `default` if you don't know what to use. It's recommended to choose Microphone and Speaker which has `Hardware device with all software conversions` notation. Set the environment variable `SATELLITE_AUDIO_DEVICE` to:

```bash
plughw:CARD=seeed2micvoicec,DEV=0
```

> `wyoming-assist-microphone` uses the same device for Mic as Speaker.

## TODO's

- [ ] Investigate whether the user should see the transcription of voice commands and responses in the Home Assistant Assist Chat popup when the name of the conversational pipeline is passed as `ASSIST_PIPELINE_NAME`.
- [ ] Split `SATELLITE_AUDIO_DEVICE` into `SATELLITE_MIC_DEVICE` and `SATELLITE_SND_DEVICE` to allow selection of different audio hardware combinations.

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

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`wyoming-assist-microphone:1.4.1`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`homeassistant-base`](/packages/smart-home/homeassistant-base) [`pip_cache`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The `wyoming-assist-microphone` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/tree/master/assist_microphone` |

| **`wyoming-assist-microphone:master`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `wyoming-assist-microphone` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`homeassistant-base`](/packages/smart-home/homeassistant-base) [`pip_cache`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The `wyoming-assist-microphone` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/tree/master/assist_microphone` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/wyoming-assist-microphone:latest-r35.4.1`](https://hub.docker.com/r/dustynv/wyoming-assist-microphone/tags) | `2024-04-30` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/wyoming-assist-microphone:latest-r36.2.0`](https://hub.docker.com/r/dustynv/wyoming-assist-microphone/tags) | `2024-04-30` | `arm64` | `0.3GB` |
| &nbsp;&nbsp;[`dustynv/wyoming-assist-microphone:r36.2.0`](https://hub.docker.com/r/dustynv/wyoming-assist-microphone/tags) | `2024-04-24` | `arm64` | `0.3GB` |

> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>
</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag wyoming-assist-microphone)

# or explicitly specify one of the container images above
jetson-containers run dustynv/wyoming-assist-microphone:latest-r35.4.1

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/wyoming-assist-microphone:latest-r35.4.1
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag wyoming-assist-microphone)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag wyoming-assist-microphone) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build wyoming-assist-microphone
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
