# piper

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<p align="center"><img src="images/piper.png" title="Wyoming piper" alt="Wyoming piper" /></p>

[`Home Assistant`](https://www.home-assistant.io/) add-on that uses [`wyoming-piper`](https://github.com/rhasspy/wyoming-piper) for text to speech system using the [`wyoming` protocol](https://www.home-assistant.io/integrations/wyoming/) on **NVIDIA Jetson** devices.

### Features

- [x] Works well with [`home-assistant-core`](packages/smart-home/homeassistant-core) container on **Jetson devices** as well as Home Assistant hosted on different host's
- [x] `GPU` Accelerated on **Jetson Devices** thank's to [`piper-tts` container](packages/audio/piper-tts)

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

  piper-tts:
    image: ms1design/wyoming-piper:master-r36.2.0-cu122-cp311
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
    environment:
      TZ: ${ENV_TZ}
    stdin_open: true
    tty: true

volumes:
  ha-config:
  ha-piper-tts-models:
```
</details>

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

## Configuration

Read more how to configure `wyoming-piper` in the [official documentation](https://www.home-assistant.io/voice_control/voice_remote_local_assistant#installing-a-local-assist-pipeline):

<p align="center"><img src="images/piper-playground.png" title="Wyoming assist-microphone" alt="Wyoming assist-microphone" /></p>

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

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`wyoming-piper:master`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `wyoming-piper` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`homeassistant-base`](/packages/smart-home/homeassistant-base) [`cuda:12.2`](/packages/cuda/cuda) [`cudnn:8.9`](/packages/cuda/cudnn) [`python:3.11`](/packages/build/python) [`tensorrt`](/packages/tensorrt) [`cmake`](/packages/build/cmake/cmake_pip) [`numpy`](/packages/numpy) [`onnx`](/packages/onnx) [`onnxruntime`](/packages/onnxruntime) [`piper-tts`](/packages/audio/piper-tts) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The `piper-tts` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/tree/master/piper` and `https://github.com/rhasspy/wyoming-piper` |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag piper)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host piper:36.2.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag piper)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag piper) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build piper
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
