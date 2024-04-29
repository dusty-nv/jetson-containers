# openwakeword

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<p align="center"><img src="images/wyoming-openwakeword.png" title="Wyoming openWakeWord" alt="Wyoming openWakeWord" /></p>

[`Home Assistant`](https://www.home-assistant.io/) add-on that uses [`openWakeWord`](https://github.com/rhasspy/wyoming-openwakeword) for wake word detection over the [`wyoming` protocol](https://www.home-assistant.io/integrations/wyoming/) on **NVIDIA Jetson** devices. Thank you to [**@ms1design**](https://github.com/ms1design) for contributing these Home Assistant & Wyoming containers!

### Features

- [x] Works well with [`home-assistant-core`](packages/smart-home/homeassistant-core) container on **Jetson devices** as well as Home Assistant hosted on different host's
- [ ] `GPU` Accelerated on **Jetson Devices** using `onnx` models [WIP] – *(For now it work's with `CPU` only utilising `tflite` models).*

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

  openwakeword:
    image: dusty-nv/wyoming-openwakeword:latest-r36.2.0-cu122-cp311
    restart: unless-stopped
    runtime: nvidia
    network_mode: host
    container_name: openwakeword
    hostname: openwakeword
    init: false
    ports:
      - "10400:10400/tcp"
    volumes:
      - ha-openwakeword-custom-models:/share/openwakeword
      - /etc/localtime:/etc/localtime:ro
      - /etc/timezone:/etc/timezone:ro
    stdin_open: true
    tty: true
    environment:
      OPENWAKEWORD_CUSTOM_MODEL_DIR: /share/openwakeword
      OPENWAKEWORD_PRELOAD_MODEL: ok_nabu

volumes:
  ha-config:
  ha-openwakeword-custom-models:
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

Read more how to configure `wyoming-openwakeword` in the [official documentation](https://www.home-assistant.io/voice_control/install_wake_word_add_on#enabling-wake-word-for-your-voice-assistant):

<p align="center"><img src="images/openwakeword-assist-config.png" title="Wyoming openWakeWord configuration" alt="Wyoming openWakeWord configuration" /></p>

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

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`wyoming-openwakeword:latest`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `wyoming-openwakeword` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`homeassistant-base`](/packages/smart-home/homeassistant-base) [`python:3.11`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The `openWakeWord` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/blob/master/openwakeword/Dockerfile` and `https://github.com/rhasspy/wyoming-openwakeword` |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag openwakeword)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host openwakeword:36.2.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag openwakeword)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag openwakeword) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build openwakeword
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
