# spark-tts

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

# SparkTTS

SparkTTS is a high-quality text-to-speech synthesis model that provides natural-sounding speech generation. This container includes the SparkTTS model running optimized for Jetson devices, offering both standard TTS functionality and zero-shot voice cloning capabilities.

## System Requirements

- **Memory**: Requires at least 5GB of available RAM

## Features

- Natural-sounding speech synthesis
- Adjustable pitch and speed
- Gender selection for standard TTS
- Zero-shot voice cloning from audio samples

## Usage Examples

When using `jetson-containers run`, the generated audio files are automatically saved in the `jetson-containers/data/audio/tts/spark-tts/` directory on your host system, and models are cached in `jetson-containers/data/models/huggingface/`.

### Standard Text-to-Speech (CLI)

Generate speech from text with customizable parameters:

```bash
jetson-containers run $(autotag spark-tts) \
    --pitch "moderate" \
    --speed "moderate" \
    --gender "female" \
    --text "The quick brown fox jumps over the lazy dog"
```

Available options:

* `--pitch`: "very_low", "low", "moderate", "high", "very_high"
* `--speed`: "very_low", "low", "moderate", "high", "very_high"
* `--gender`: "female", "male"

### Zero-shot Voice Cloning (CLI) 

Clone a voice from a sample audio file (note: the audio file must be accessible inside the container, put it in the jetson-containers/data directory):

```bash
jetson-containers run $(autotag spark-tts) \
    --prompt_speech_path "/data/audio/sample.wav" \
    --prompt_text "This is a sample prompt text that matches the audio sample..." \
    --speed "moderate" \
    --text "Hi, this is a test of voice cloning with Spark TTS!"
```

## Output Location

When using `jetson-containers run`, the following directories are automatically mounted and accessible:

* Audio output: `jetson-containers/data/audio/tts/spark-tts/`
* Model cache: `jetson-containers/data/models/huggingface/`

The generated audio files will be saved with timestamped filenames like `20250325230742.wav`.


## Model Source

This container uses the SparkTTS model from Hugging Face:
[Spark-TTS by SparkAudio](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`spark-tts`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchaudio`](/packages/pytorch/torchaudio) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Notes | Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens – https://github.com/SparkAudio/Spark-TTS |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag spark-tts)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host spark-tts:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag spark-tts)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag spark-tts) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build spark-tts
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
