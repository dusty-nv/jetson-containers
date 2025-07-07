# kokoro-tts-fastapi

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

# KokoroTTS FastAPI

This Dockerfile sets up a text-to-speech (TTS) service based on the Kokoro TTS engine using FastAPI (https://github.com/remsky/Kokoro-FastAPI)

The server will automatically be started at `0.0.0.0:8880` when you run the container:

```
docker run -it --rm -p 8880:8880 dustynv/kokoro-tts:fastapi-r36.4.0
```

Included is a built-in web UI ([`http://0.0.0.0:8880/web`](http://0.0.0.0:8880/web)) that converts text to speech using the Kokoro TTS model with GPU acceleration.  

> [!WARNING]  
> There appears to be an issue with audio playback in the web UI on Firefox - Chrome or Chromium is recommended. 

This is the default CMD executed at container startup:

```
uvicorn api.src.main:app --reload --host 0.0.0.0 --port 8880
```
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`kokoro-tts:fastapi`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`torchaudio`](/packages/pytorch/torchaudio) [`kokoro-tts:hf`](/packages/speech/kokoro-tts/kokoro-tts-hf) [`ffmpeg`](/packages/multimedia/ffmpeg) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/kokoro-tts:fastapi-r36.4.0`](https://hub.docker.com/r/dustynv/kokoro-tts/tags) `(2025-02-19, 5.4GB)`<br>[`dustynv/kokoro-tts:fastapi-r36.4.0-cu128-24.04`](https://hub.docker.com/r/dustynv/kokoro-tts/tags) `(2025-03-03, 4.8GB)` |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag kokoro-tts-fastapi)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host kokoro-tts-fastapi:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag kokoro-tts-fastapi)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag kokoro-tts-fastapi) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build kokoro-tts-fastapi
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
