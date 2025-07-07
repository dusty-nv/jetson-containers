# speaches

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

# Speaches

Speaches supports VAD (Voice Activity Detection), STT (Speech-to-Text) and TTS (Text-to-Speech).

## What is VAD and why it's needed

Voice Activity Detection (VAD) is used to detect the presence or absence of human speech in audio streams.
It is important when processing speech because by identifying when someone is actually speaking, you prevent unnecessary processing of silence or background noise, reducing computational overhead.

## Important Notes on TTS Implementation

At this time of adding support to Speaches, Kokoro uses `onnxruntime` and `kokoro-onnx` for TTS.

This backend is currently slower than `kokoro-tts:fastapi`, which also implements the OpenAI protocol, and can be used separately.


## Build

```bash
CUDA_VERSION=12.9 LSB_RELEASE=24.04 PYTHON_VERSION=3.12 jc build speaches
```

or

```bash
CUDA_VERSION=12.9 LSB_RELEASE=24.04 PYTHON_VERSION=3.12  jetson-containers build --build-args=PIP_RETRIES:10,PIP_TIMEOUT:60 speaches
```

## Run

Run it with:
```
docker run -it --rm --network=host <docker-image-name>
```


Speaches site is at:
```
http://localhost:8000/
```



## Contributing

If you need TTS endpoint support, please open an issue and tag @OriNachum


<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`speaches`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`cmake`](/packages/build/cmake/cmake_pip) [`ctranslate2`](/packages/ml/ctranslate2) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`onnx`](/packages/ml/onnx) [`onnxruntime`](/packages/ml/onnxruntime) [`huggingface_hub`](/packages/llm/huggingface_hub) [`faster-whisper`](/packages/speech/faster-whisper) [`pytorch:2.8`](/packages/pytorch) [`torchaudio`](/packages/pytorch/torchaudio) [`espeak`](/packages/speech/espeak) [`piper-tts`](/packages/speech/piper-tts) [`numba`](/packages/numeric/numba) [`kokoro-tts:onnx`](/packages/speech/kokoro-tts/kokoro-tts-onnx) [`nodejs`](/packages/build/nodejs) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/speaches:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/speaches/tags) `(2025-03-03, 6.6GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/speaches:r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/speaches/tags) | `2025-03-03` | `arm64` | `6.6GB` |

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
jetson-containers run $(autotag speaches)

# or explicitly specify one of the container images above
jetson-containers run dustynv/speaches:r36.4.0-cu128-24.04

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/speaches:r36.4.0-cu128-24.04
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag speaches)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag speaches) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build speaches
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
