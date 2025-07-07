# chatterbox-tts

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

# Chatterbox TTS

Chatterbox TTS is an advanced text-to-speech engine that offers high-quality voice synthesis with emotion control and voice cloning capabilities. It's based on the [Chatterbox project by Resemble AI](https://github.com/resemble-ai/chatterbox).

## Overview

Chatterbox TTS is a more resource-intensive model compared to alternatives like Piper and Kokoro, offering enhanced synthesis quality and additional features. The added capabilities come at the cost of higher computational requirements.

## Features

- **Standard Voice Generation**: Generate natural-sounding speech from text
- **Emotional Voice Synthesis**: Control the emotional tone of generated speech
- **Voice Cloning**: Clone voices from audio samples for personalized speech output

## Hardware Compatibility

| Device | Status |
|--------|--------|
| Jetson AGX | ✅ Tested |
| Jetson Nano | 🔄 Testing in progress |
| Other Jetson devices | 📝 To be tested |

## System Requirements

- CUDA 12.9+
- Ubuntu 24.04 or compatible OS
- Sufficient GPU memory (tested on Jetson AGX)

## Performance

- **Claimed inference time:** ~200ms per generation
- **Actual benchmarks:** Pending detailed testing across Jetson devices

## Usage

### Building the container

When building the container, use:
```bash
CUDA_VERSION=12.9 LSB_RELEASE=24.04 jetson-containers build --name=... chatterbox-tts 
```


### Running the Container

```bash
docker run -it --rm --runtime nvidia dustynv/chatterbox-tts:r36.2.0
```

### Basic Text-to-Speech Generation

```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Load model to GPU
model = ChatterboxTTS.from_pretrained(device="cuda")

# Generate speech
text = "Hello, I am Chatterbox TTS running on NVIDIA Jetson!"
wav = model.generate(text)
ta.save("output.wav", wav, model.sr)
```

### Voice Cloning

```python
# Load reference audio for voice cloning
reference_audio = "path/to/reference.wav"
wav = model.generate(text, reference_audio_path=reference_audio)
ta.save("cloned_voice.wav", wav, model.sr)
```

## ⚠️ Development Status: Work in Progress ⚠️

This package is currently under active development. Planned improvements include:

- Proper Docker container lifecycle management
- Comprehensive test suite
- Extended device compatibility testing
- Performance optimizations for Jetson devices
- Expanded documentation with more usage examples

## Troubleshooting

If you encounter issues with GPU memory or performance:
- Ensure you have sufficient GPU memory available
- Consider reducing batch size or sequence length for larger inputs
- Check that you're using the appropriate CUDA version

## Additional Resources

- [Chatterbox GitHub Repository](https://github.com/resemble-ai/chatterbox)
- [Jetson Containers Documentation](https://github.com/dusty-nv/jetson-containers)


<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`chatterbox-tts`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`torch`](/packages/pytorch) [`pytorch:2.8`](/packages/pytorch) [`torchaudio`](/packages/pytorch/torchaudio) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`diffusers`](/packages/diffusion/diffusers) [`sound-utils`](/packages/multimedia/sound-utils) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag chatterbox-tts)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host chatterbox-tts:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag chatterbox-tts)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag chatterbox-tts) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build chatterbox-tts
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
