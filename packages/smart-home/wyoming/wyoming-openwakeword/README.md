# wyoming-openwakeword

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<p align="center"><img src="images/wyoming-openwakeword.png" title="Wyoming openWakeWord" alt="Wyoming openWakeWord" style="width:100%;max-width:600px" /></p>

[`Home Assistant`](https://www.home-assistant.io/) add-on that uses [`openWakeWord`](https://github.com/rhasspy/wyoming-openwakeword) ([demo on huggingface](https://huggingface.co/spaces/davidscripka/openWakeWord)) for wake word detection over the [`wyoming` protocol](https://www.home-assistant.io/integrations/wyoming/) on **NVIDIA Jetson** devices. Thank you to [**@ms1design**](https://github.com/ms1design) for contributing these Home Assistant & Wyoming containers!

## Features

- [x] Works well with [`home-assistant-core`](/packages/smart-home/homeassistant-core) container on **Jetson devices** as well as Home Assistant hosted on different hosts
- [x] Use [custom wake word's](#training-custom-wake-words), pass model name as `OPENWAKEWORD_PRELOAD_MODEL` to preload custom model. For example you can find `jetson` (*`jets_un`*) wake word model included in `/share/openwakeword` models directory.
- [x] Supports `*.tflite` `CPU` wake word models
- [ ] Supports `*.onnx` `CUDA` wake word models [WIP]

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

  openwakeword:
    image: dustynv/wyoming-openwakeword:latest-r36.2.0
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
    environment:
      OPENWAKEWORD_CUSTOM_MODEL_DIR: /share/openwakeword
      OPENWAKEWORD_PRELOAD_MODEL: ok_nabu

volumes:
  ha-config:
  ha-openwakeword-custom-models:
```

## Environment variables

| Variable | Type | Default | Description
| - | - | - | - |
| `OPENWAKEWORD_PORT` | `str` | `10400` | Port number to use on `host` |
| `OPENWAKEWORD_THRESHOLD` | `float` | `0.5` | Wake word model threshold (`0.0`-`1.0`), where higher means fewer activations. |
| `OPENWAKEWORD_TRIGGER_LEVEL` | `int` | `1` | Number of activations before a detection is registered. A higher trigger level means fewer detections. |
| `OPENWAKEWORD_PRELOAD_MODEL`| `str` | `ok_nabu` | Name or path of wake word model to pre-load. The name of the model should match with name used during [custom wake word model training](#training-custom-wake-words). When changing this, it's also recommended to set `WAKEWORD_NAME` variable with same value for [`wyoming-assist-microphone`](/packages/smart-home/wyoming/assist-microphone) container |
| `OPENWAKEWORD_CUSTOM_MODEL_DIR` | `str` | `/share/openwakeword` | Path to directory containing custom wake word models. *Skip the trailing slash (`/`)* |
| `OPENWAKEWORD_DEBUG` | `bool` | `true` | Log `DEBUG` messages |

## Configuration

Read more how to configure `wyoming-openwakeword` in the [official documentation](https://www.home-assistant.io/voice_control/install_wake_word_add_on#enabling-wake-word-for-your-voice-assistant):

<p align="center"><img src="images/openwakeword-assist-config.png" title="Wyoming openWakeWord configuration" alt="Wyoming openWakeWord configuration" style="width:100%;max-width:600px" /></p>

## Training custom wake word's

> [!NOTE]
> You can find a custom trained, example `jetson` (`jets_un`) wake word model in the custom models directory (`/share/openwakeword`). To use it, set `WAKEWORD_NAME` to `jets_un` in appropriate containers.

The Home Assistant Community has [trained numerous wake word models](https://github.com/fwartner/home-assistant-wakewords-collection), as detailed in this GitHub repository. However, these models are specifically designed for use with `CPU`.

To train a new wake word model for `CPU` (`*.tflite`) or `cuda` (`*.onnx`), you can follow [@dscripka](https://github.com/dscripka) [documentation](https://github.com/dscripka/openWakeWord?tab=readme-ov-file#training-new-models) or just jump to the point and use [wake word training environment](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb).

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

> [!NOTE]
> This project was created by [Jetson AI Lab Research Group](https://www.jetson-ai-lab.com/research.html).
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`wyoming-openwakeword:1.10.0`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`homeassistant-base`](/packages/smart-home/homeassistant-base) [`pip_cache`](/packages/cuda/cuda) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The `openWakeWord` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/blob/master/openwakeword/Dockerfile` and `https://github.com/rhasspy/wyoming-openwakeword` |

| **`wyoming-openwakeword:master`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `wyoming-openwakeword` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`homeassistant-base`](/packages/smart-home/homeassistant-base) [`pip_cache`](/packages/cuda/cuda) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | The `openWakeWord` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/blob/master/openwakeword/Dockerfile` and `https://github.com/rhasspy/wyoming-openwakeword` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/wyoming-openwakeword:latest-r36.2.0`](https://hub.docker.com/r/dustynv/wyoming-openwakeword/tags) | `2024-04-30` | `arm64` | `0.3GB` |
| &nbsp;&nbsp;[`dustynv/wyoming-openwakeword:r35.4.1`](https://hub.docker.com/r/dustynv/wyoming-openwakeword/tags) | `2024-04-10` | `arm64` | `5.1GB` |
| &nbsp;&nbsp;[`dustynv/wyoming-openwakeword:r36.2.0`](https://hub.docker.com/r/dustynv/wyoming-openwakeword/tags) | `2024-04-24` | `arm64` | `0.3GB` |

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
jetson-containers run $(autotag wyoming-openwakeword)

# or explicitly specify one of the container images above
jetson-containers run dustynv/wyoming-openwakeword:latest-r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/wyoming-openwakeword:latest-r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag wyoming-openwakeword)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag wyoming-openwakeword) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build wyoming-openwakeword
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
