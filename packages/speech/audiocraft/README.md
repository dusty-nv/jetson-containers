# audiocraft

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

docs.md
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`audiocraft`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=34.1.0']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) [`torchtext`](/packages/pytorch/torchtext) [`opengl`](/packages/multimedia/opengl) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`triton`](/packages/ml/triton) [`diffusers`](/packages/diffusion/diffusers) [`xformers`](/packages/attention/xformers) [`tensorrt`](/packages/cuda/tensorrt) [`cuda-python`](/packages/cuda/cuda-python) [`pycuda`](/packages/cuda/pycuda) [`jupyterlab`](/packages/code/jupyterlab) |
| &nbsp;&nbsp;&nbsp;Dependants | [`voice-pro`](/packages/speech/voice-pro) [`voicecraft`](/packages/speech/voicecraft) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/audiocraft:r35.2.1`](https://hub.docker.com/r/dustynv/audiocraft/tags) `(2023-11-05, 10.7GB)`<br>[`dustynv/audiocraft:r35.3.1`](https://hub.docker.com/r/dustynv/audiocraft/tags) `(2024-03-07, 7.1GB)`<br>[`dustynv/audiocraft:r35.4.1`](https://hub.docker.com/r/dustynv/audiocraft/tags) `(2024-01-09, 7.0GB)`<br>[`dustynv/audiocraft:r36.2.0`](https://hub.docker.com/r/dustynv/audiocraft/tags) `(2024-05-02, 6.6GB)`<br>[`dustynv/audiocraft:r36.3.0`](https://hub.docker.com/r/dustynv/audiocraft/tags) `(2024-06-08, 6.6GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/audiocraft:r35.2.1`](https://hub.docker.com/r/dustynv/audiocraft/tags) | `2023-11-05` | `arm64` | `10.7GB` |
| &nbsp;&nbsp;[`dustynv/audiocraft:r35.3.1`](https://hub.docker.com/r/dustynv/audiocraft/tags) | `2024-03-07` | `arm64` | `7.1GB` |
| &nbsp;&nbsp;[`dustynv/audiocraft:r35.4.1`](https://hub.docker.com/r/dustynv/audiocraft/tags) | `2024-01-09` | `arm64` | `7.0GB` |
| &nbsp;&nbsp;[`dustynv/audiocraft:r36.2.0`](https://hub.docker.com/r/dustynv/audiocraft/tags) | `2024-05-02` | `arm64` | `6.6GB` |
| &nbsp;&nbsp;[`dustynv/audiocraft:r36.3.0`](https://hub.docker.com/r/dustynv/audiocraft/tags) | `2024-06-08` | `arm64` | `6.6GB` |

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
jetson-containers run $(autotag audiocraft)

# or explicitly specify one of the container images above
jetson-containers run dustynv/audiocraft:r36.3.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/audiocraft:r36.3.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag audiocraft)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag audiocraft) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build audiocraft
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
