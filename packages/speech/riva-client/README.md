# riva-client

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


* these are the NVIDIA Riva [C++](https://github.com/nvidia-riva/cpp-clients) and [Python](https://github.com/nvidia-riva/python-clients) clients only (found under `/opt/riva/python-clients`)
* see [`riva_quickstart_arm64`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64) from NGC to start the core Riva server container first
* Riva API reference docs:  https://docs.nvidia.com/deeplearning/riva/user-guide/docs/

### Start Riva Server

Before doing anything, you should download and run the Riva server container from [`riva_quickstart_arm64`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64) using `riva_start.sh`

This will run locally on your Jetson Xavier or Orin device and is [supported on JetPack 5](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/support-matrix.html#embedded).  You can disable NLP/NMT in its `config.sh` and it will use around ~5GB of memory for ASR+TTS.  It's then recommended to test the system with [these examples](https://github.com/nvidia-riva/python-clients#asr) under `/opt/riva/python-clients`

You can also see this helpful video and guide from JetsonHacks for setting up Riva:  [**Speech AI on Jetson Tutorial**](https://jetsonhacks.com/2023/08/07/speech-ai-on-nvidia-jetson-tutorial/)

### List Audio Devices

This will print out a list of audio input/output devices that are connected to your system:

```bash
./run.sh --workdir /opt/riva/python-clients $(./autotag riva-client:python) \
   python3 scripts/list_audio_devices.py
```

You can refer to them in the steps below by either their device number or name.  Depending on the sample rate they support, you may also need to set `--sample-rate-hz` below to a valid frequency (e.g. `16000` `44100` `48000`)

### Streaming ASR

```bash
./run.sh --workdir /opt/riva/python-clients $(./autotag riva-client:python) \
   python3 scripts/asr/transcribe_mic.py --input-device=24 --sample-rate-hz=48000
```

You can find more ASR examples to run at https://github.com/nvidia-riva/python-clients#asr

### Streaming TTS

```bash
./run.sh --workdir /opt/riva/python-clients $(./autotag riva-client:python) \
   python3 scripts/tts/talk.py --stream --output-device=24 --sample-rate-hz=48000 \
     --text "Hello, how are you today? My name is Riva." 
```

You can set the `--voice` argument to one of the [available voices](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html#voices) (the default is `English-US.Female-1`)

Also, you can customize the rate, pitch, and pronunciation of individual words/phrases by including [inline SSML](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/tts-basics-customize-ssml.html#customizing-riva-tts-audio-output-with-ssml) in your text.

### Loopback

To feed the live ASR transcript into the TTS and have it speak your words back to you:

```bash
./run.sh --workdir /opt/riva/python-clients $(./autotag riva-client:python) \
   python3 scripts/loopback.py --input-device=24 --output-device=24 --sample-rate-hz=48000
```
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`riva-client:cpp`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`bazel`](/packages/build/bazel) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.cpp`](Dockerfile.cpp) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/riva-client:cpp-r35.2.1`](https://hub.docker.com/r/dustynv/riva-client/tags) `(2023-08-29, 6.3GB)`<br>[`dustynv/riva-client:cpp-r35.3.1`](https://hub.docker.com/r/dustynv/riva-client/tags) `(2024-02-24, 6.3GB)`<br>[`dustynv/riva-client:cpp-r35.4.1`](https://hub.docker.com/r/dustynv/riva-client/tags) `(2023-10-07, 6.3GB)` |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/nvidia-riva/cpp-clients |

| **`riva-client:python`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache`](/packages/cuda/cuda) [`python`](/packages/build/python) |
| &nbsp;&nbsp;&nbsp;Dependants | [`llamaspeak`](/packages/llm/llamaspeak) [`local_llm`](/packages/llm/local_llm) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile.python`](Dockerfile.python) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/riva-client:python-r35.2.1`](https://hub.docker.com/r/dustynv/riva-client/tags) `(2023-09-07, 5.0GB)`<br>[`dustynv/riva-client:python-r35.3.1`](https://hub.docker.com/r/dustynv/riva-client/tags) `(2024-02-24, 5.0GB)`<br>[`dustynv/riva-client:python-r35.4.1`](https://hub.docker.com/r/dustynv/riva-client/tags) `(2023-10-07, 5.0GB)`<br>[`dustynv/riva-client:python-r36.2.0`](https://hub.docker.com/r/dustynv/riva-client/tags) `(2024-03-11, 0.3GB)` |
| &nbsp;&nbsp;&nbsp;Notes | https://github.com/nvidia-riva/python-clients |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/riva-client:cpp-r35.2.1`](https://hub.docker.com/r/dustynv/riva-client/tags) | `2023-08-29` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/riva-client:cpp-r35.3.1`](https://hub.docker.com/r/dustynv/riva-client/tags) | `2024-02-24` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/riva-client:cpp-r35.4.1`](https://hub.docker.com/r/dustynv/riva-client/tags) | `2023-10-07` | `arm64` | `6.3GB` |
| &nbsp;&nbsp;[`dustynv/riva-client:python-r35.2.1`](https://hub.docker.com/r/dustynv/riva-client/tags) | `2023-09-07` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/riva-client:python-r35.3.1`](https://hub.docker.com/r/dustynv/riva-client/tags) | `2024-02-24` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/riva-client:python-r35.4.1`](https://hub.docker.com/r/dustynv/riva-client/tags) | `2023-10-07` | `arm64` | `5.0GB` |
| &nbsp;&nbsp;[`dustynv/riva-client:python-r36.2.0`](https://hub.docker.com/r/dustynv/riva-client/tags) | `2024-03-11` | `arm64` | `0.3GB` |
| &nbsp;&nbsp;[`dustynv/riva-client:r35.2.1`](https://hub.docker.com/r/dustynv/riva-client/tags) | `2023-08-10` | `arm64` | `6.3GB` |

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
jetson-containers run $(autotag riva-client)

# or explicitly specify one of the container images above
jetson-containers run dustynv/riva-client:python-r36.2.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/riva-client:python-r36.2.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag riva-client)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag riva-client) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build riva-client
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
